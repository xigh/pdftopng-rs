use base64::Engine;
use log::{debug, info, trace};
use std::{path::Path, time::Instant};

use anyhow::Result;
use clap::Parser;
use futures_util::TryStreamExt;
use pdfium_render::prelude::*;
use progress_bar::{
    Color, Style, finalize_progress_bar, inc_progress_bar, init_progress_bar,
    set_progress_bar_action,
};

mod args;
use args::Args;

mod ollama;
use ollama::{ChatMessage, GenerateOptions, OllamaClient, Role};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::try_parse()?;

    env_logger::builder()
        .filter_level(args.log_level.parse().unwrap())
        .init();

    let ollamas = args
        .ollama_url
        .iter()
        .map(|url| {
            let (url, count) = url.split_once('@').unwrap_or((url, "1"));
            let count = count.parse::<usize>().unwrap_or(1);
            println!("Creating {} ollamas from {:?}", count, url);
            OllamaClient::new(url, &args.model, count)
        })
        .collect::<Vec<_>>();

    if args.enum_models && !args.ollama_url.is_empty() {
        for ollama in ollamas {
            println!("Listing models from {}", ollama.url());
            let mut models = ollama.list_models().await?;

            let sfx2scale = |sfx: char| match sfx {
                'B' => Some(1_000_000_000.0),
                'M' => Some(1_000_000.0),
                'K' => Some(1_000.0),
                _ => None,
            };

            models.sort_by(|a, b| {
                if args.sort_by_size {
                    let a_details = a.details.clone().unwrap_or(serde_json::Value::default());
                    let a_parameter_size = a_details
                        .get("parameter_size")
                        .unwrap_or_default()
                        .as_str()
                        .unwrap_or_default();
                    let a_sfx = a_parameter_size.chars().last().unwrap_or_default();
                    let a_scale = sfx2scale(a_sfx).unwrap();
                    let a_trimmed = a_parameter_size.trim_end_matches(a_sfx);
                    let a_size = a_trimmed.parse::<f64>().unwrap_or_default();

                    let b_details = b.details.clone().unwrap_or(serde_json::Value::default());
                    let b_parameter_size = b_details
                        .get("parameter_size")
                        .unwrap_or_default()
                        .as_str()
                        .unwrap_or_default();
                    let b_sfx = b_parameter_size.chars().last().unwrap_or_default();
                    let b_scale = sfx2scale(b_sfx).unwrap();
                    let b_trimmed = b_parameter_size.trim_end_matches(b_sfx);
                    let b_size = b_trimmed.parse::<f64>().unwrap_or_default();

                    (a_size * a_scale)
                        .partial_cmp(&(b_size * b_scale))
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.name.clone().cmp(&b.name.clone())
                }
            });

            for model in models {
                let details = model.details.unwrap_or(serde_json::Value::default());
                let parameter_size = details
                    .get("parameter_size")
                    .unwrap_or_default()
                    .as_str()
                    .unwrap_or_default();
                let quantization_level = details
                    .get("quantization_level")
                    .unwrap_or_default()
                    .as_str()
                    .unwrap_or_default();
                println!(
                    " - {:<-40} {:>10} {:>10}",
                    model.name, parameter_size, quantization_level
                );
            }
        }
        return Ok(());
    }

    let mut ollama_list = Vec::new();
    for ollama in &ollamas {
        let ollama_url = ollama.url().to_string();
        let ollama_count = ollama.count();
        println!("Adding {} ollamas from {:?}", ollama_count, ollama_url);
        for _ in 0..ollama_count {
            ollama_list.push(ollama);
        }
    }

    let pdfium = Pdfium::default();

    let start = Instant::now();
    for input_pdf in args.files {
        let input_file = Path::new(&input_pdf).file_name().unwrap().to_str().unwrap();
        println!("Loading {}", input_file);

        let document = pdfium.load_pdf_from_file(&input_pdf, None)?;
        if args.verbose {
            println!("Document {:?} chargé en {:?}", input_pdf, start.elapsed());
        }

        let page_count = document.pages().len();
        let page_start = args.page_start.unwrap_or(1);
        if page_start == 0 {
            return Err(anyhow::anyhow!("Page start cannot be 0"));
        }
        let page_end = args.page_end.unwrap_or(page_count as usize);
        if page_end < page_start {
            return Err(anyhow::anyhow!("Page end cannot be less than page start"));
        }
        if page_end > page_count as usize {
            return Err(anyhow::anyhow!(
                "Page end cannot be greater than page count"
            ));
        }

        init_progress_bar(page_end - page_start + 1);

        let dir_path = Path::new("output");
        std::fs::create_dir_all(dir_path).unwrap();

        let mut pages_to_remove = Vec::new();
        let mut handles = Vec::new();

        let start = Instant::now();
        let pages = document.pages();
        for (page_no, page) in pages.iter().enumerate() {
            let page_no = page_no + 1;
            if page_no < page_start {
                continue;
            }
            if page_no > page_end {
                break;
            }

            set_progress_bar_action("processing", Color::Green, Style::Bold);

            if args.show_content {
                for object in page.objects().iter() {
                    if let Some(text_object) = object.as_text_object() {
                        let h = text_object.get_horizontal_translation();
                        let v = text_object.get_vertical_translation();
                        println!(
                            "Content: {:?} [{:?},{:?}]",
                            text_object.text(),
                            h.to_mm(),
                            v.to_mm()
                        );
                    }
                }
            }

            let bitmap = page.render_with_config(
                &PdfRenderConfig::new().set_target_width(args.page_width.into()),
            )?;

            // convert to rgba8
            let width = bitmap.width() as u32;
            let height = bitmap.height() as u32;
            let image = bitmap.as_image();
            let rgba = image.as_rgba8().unwrap();

            let base_input_pdf = Path::new(&input_pdf).file_name().unwrap().to_str().unwrap();

            // write to png
            let page_path =
                base_input_pdf.replace(".pdf", format!("-page-{:06}.png", page_no).as_str());
            let image_path = dir_path.join(page_path);

            // write to memory buffer first
            let mut buffer = Vec::new();
            let mut encoder = png::Encoder::new(&mut buffer, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);

            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgba).unwrap();
            writer.finish().unwrap();

            // write buffer to file
            std::fs::write(&image_path, &buffer).unwrap();

            // encode to base64
            let base64 = base64::engine::general_purpose::STANDARD.encode(&buffer);

            let chat_message = ChatMessage {
                role: Role::User,
                content: args.prompt.clone(),
                thinking: None,
                images: Some(vec![base64]),
            };
            let messages = vec![chat_message];

            let options = GenerateOptions {
                temperature: Some(0.0),
                top_p: None,
                top_k: None,
                num_predict: None,
            };

            let ollama = &ollama_list[(page_no - 1) % ollama_list.len()];
            let ollama_url = ollama.url().to_string();

            println!("Sending request to Ollama {:?}", ollama_url);
            let mut stream = ollama.generate_stream(&messages, &options);
            let content_name =
                base_input_pdf.replace(".pdf", format!("-page-{:06}.md", page_no).as_str());
            let content_path = dir_path.join(content_name);

            let handle = tokio::spawn(async move {
                let mut token_count = 0;
                let mut accumulated_response = String::new();
                let mut start = None;
                while let Some(response) = stream.try_next().await.unwrap() {
                    if start.is_none() {
                        start = Some(Instant::now());
                    }
                    trace!("Response: {:?}", response);
                    debug!(
                        "Processing response: done={}, text={}",
                        response.done, response.message.content
                    );
                    accumulated_response += &response.message.content;
                    token_count += response.message.content.len();
                    if token_count > args.max_tokens {
                        info!("Max tokens reached, stopping stream");
                        break;
                    }
                }
                println!(
                    " - page {} {:?}, {} tokens in {:?}",
                    page_no,
                    ollama_url,
                    token_count,
                    start.unwrap().elapsed()
                );

                std::fs::write(&content_path, accumulated_response).unwrap();
            });
            handles.push(handle);

            pages_to_remove.push(image_path);
        }

        for handle in handles {
            inc_progress_bar();
            handle.await.unwrap();
        }
        finalize_progress_bar();

        println!("{} processed in {:?}", input_file, start.elapsed());

        if !args.keep {
            for page in pages_to_remove {
                std::fs::remove_file(page).unwrap();
            }
        }
    }

    Ok(())
}
