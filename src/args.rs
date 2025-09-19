use clap::{Parser, arg, ValueHint};

const DEFAULT_PROMPT: &str = r"
Task: Transcribe the page from the provided book image.

- Reproduce the text exactly as it appears, without adding or omitting anything.
- Do not interpret the text, just transcribe it exactly as it appears.
- Use Markdown syntax to preserve the original formatting (e.g., headings, bold, italics, lists).
- Do not include triple backticks or any other code block markers in your response, unless the page contains code.
- Do not add any headers, topics or footers, such as `**Heading**` or `**Bullet points**`, keep just raw text.
- If the page contains an image, or a diagram, describe it in detail. Enclose the description in an <image> tag. For example:

<image>
This is an image of a cat.
</image>
";


#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,

    #[arg(short = 'c', long = "show-content")]
    pub show_content: bool,

    #[arg(short = 'l', long, default_value = "error")]
    pub log_level: String,

    #[arg(short = 'w', long, default_value = "1600")]
    pub page_width: u16,

    #[arg(short = 'k', long)]
    pub keep: bool, // keep pages

    #[arg(short = 's', long)]
    pub page_start: Option<usize>,

    #[arg(short = 'e', long)]
    pub page_end: Option<usize>,

    #[arg(short = 'o', long, default_value = "output")]
    pub output_dir: String,

    #[arg(long = "ls")]
    pub enum_models: bool,

    #[arg(long = "sort-by-size", default_value = "false")]
    pub sort_by_size: bool,

    #[arg(short = 'u', long, default_value = "http://localhost:11434", value_delimiter = ',')]
    pub ollama_url: Vec<String>,

    #[arg(long = "prompt", default_value = DEFAULT_PROMPT)]
    pub prompt: String,

    #[arg(short = 'm', long, default_value = "qwen2.5vl:latest")]
    pub model: String,

    #[arg(long = "max-tokens", default_value = "1024")]
    pub max_tokens: usize,

    #[arg(value_name = "FILES", num_args = 1.., value_hint = ValueHint::FilePath)]
    pub files: Vec<String>,
}
