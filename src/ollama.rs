use anyhow::Result;
use futures_util::{TryStreamExt, stream::Stream};
use log::{error, debug, trace, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role, // system, user, assistant, or tool
    pub content: String,
    pub thinking: Option<String>,
    pub images: Option<Vec<String>>,
	// tool_calls: []ToolCall  `json:"tool_calls,omitempty"`
	// tool_name:  string      `json:"tool_name,omitempty"`
}

// type ToolCall struct {
// 	Function ToolCallFunction `json:"function"`
// }

// type ToolCallFunction struct {
// 	Index     int                       `json:"index,omitempty"`
// 	Name      string                    `json:"name"`
// 	Arguments ToolCallFunctionArguments `json:"arguments"`
// }

// type ToolCallFunctionArguments map[string]any


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub num_predict: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub options: GenerateOptions,
    pub stream: bool,
    /*
    "format": {
        "type": "object",
        "properties": {
            "age": {
                "type": "integer"
            },
            "available": {
                "type": "boolean"
            }
        },
        "required": [
            "age",
            "available"
        ]
    },
    */
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: i64,
    pub digest: String,
    pub details: Option<Value>,
}

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum OllamaError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    count: usize,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str, count: usize) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            count,
        }
    }

    #[allow(unused)]
    pub fn url(&self) -> &str {
        &self.base_url
    }

    #[allow(unused)]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[allow(unused)]
    pub fn count(&self) -> usize {
        self.count
    }

    #[allow(unused)]
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let client = Client::new();
        let url = format!("{}/api/tags", self.base_url);

        debug!("Listing models from: {}", url);

        let response = client.get(&url).send().await?;

        debug!("Response status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await?;
            error!("Error response body: {}", error_body);
            return Err(anyhow::anyhow!("Ollama API error: {}", status));
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }

        let response_text = response.text().await?;
        trace!("Response: {}", response_text);

        let response: ModelsResponse = serde_json::from_str(&response_text)?;
        debug!("Found {} models", response.models.len());
        for model in &response.models {
            debug!("- {} ({} bytes)", model.name, model.size);
        }

        Ok(response.models)
    }

    pub fn generate_stream(
        &self,
        messages: &Vec<ChatMessage>,
        options: &GenerateOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<OllamaResponse>> + Send>> {
        let client = Client::new();
        let url = format!("{}/api/chat", self.base_url.clone());
        let model = self.model.clone();
        let messages = messages.clone();
        let options = options.clone();
    
        let fut = async_stream::try_stream! {
            let request = GenerateRequest {
                model,
                messages,
                options,
                stream: true,
            };
    
            let resp = client
                .post(&url)
                .header("Accept", "application/x-ndjson") // pas obligatoire mais explicite
                .json(&request)
                .send()
                .await?
                .error_for_status()?;
    
            // Récupère un flux de chunks (Bytes)
            let mut stream = resp.bytes_stream();
    
            // Buffer pour gérer les JSON splités sur plusieurs chunks
            let mut buf = String::new();
    
            while let Some(chunk) = stream.try_next().await? {
                // Append le chunk courant au buffer
                let s = String::from_utf8_lossy(&chunk);
                buf.push_str(&s);
    
                // On traite toutes les lignes complètes disponibles
                let mut start = 0usize;
                while let Some(nl_pos) = buf[start..].find('\n') {
                    let end = start + nl_pos;
                    let line = buf[start..end].trim();
                    if !line.is_empty() {
                        match serde_json::from_str::<OllamaResponse>(line) {
                            Ok(msg) => {
                                // On émet l'élément streamé
                                yield msg;
                            }
                            Err(e) => {
                                // Si ça échoue ici, c'est probablement qu'on n'avait pas une ligne complète.
                                // Mais comme on a trouvé un '\n', on log pour debug.
                                debug!("JSON line parse error (will keep buffering): {e}; line=`{line}`");
                            }
                        }
                    }
                    // on avance après ce '\n'
                    start = end + 1;
                }
    
                // Conserve le reste partiel (après le dernier '\n') dans buf
                if start > 0 {
                    buf.drain(..start);
                }
            }
    
            // Fin du flux HTTP : s'il reste quelque chose dans le buffer sans '\n', tente un dernier parse
            let tail = buf.trim();
            if !tail.is_empty() {
                if let Ok(msg) = serde_json::from_str::<OllamaResponse>(tail) {
                    yield msg;
                } else {
                    debug!("Trailing partial JSON not parsed: `{tail}`");
                }
            }
        };
    
        Box::pin(fut)
    }

    #[allow(unused)]
    pub fn generate_stream_old(&self, messages: &Vec<ChatMessage>, options: &GenerateOptions) -> Pin<Box<dyn Stream<Item = Result<OllamaResponse>> + Send>> {
        let client = Client::new();
        let url = format!("{}/api/chat", self.base_url.clone());
        let model = self.model.clone();
        let messages = messages.clone();
        let options = options.clone();

        info!("Sending request to Ollama at: {}", url);
        info!("Using model: {}", model);
        info!("Options: {:?}", options);
        debug!("Prompt: {:?}", messages);

        let fut = async_stream::try_stream! {
            let request = GenerateRequest {
                model,
                messages,
                options,
                stream: true,
            };
            debug!("request: {:?}", request);
            let response = client
                .post(&url)
                .json(&request)
                .send()
                .await?;
            let response = response.error_for_status().map_err(|e| {
                anyhow::anyhow!("Ollama API error: {}", e)
            })?;
            debug!("response: {:?}", response);
            let bytes = response.bytes().await?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let mut buffer = text;
            let mut new_buffer = String::new();
            for line in buffer.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                debug!("line: {}", line);
                match serde_json::from_str::<OllamaResponse>(line) {
                    Ok(response) => {
                        yield response;
                    }
                    Err(err) => {
                        error!("Error parsing line: {}", err);
                        new_buffer.push_str(line);
                        new_buffer.push('\n');
                    }
                }
            }
            buffer = new_buffer;
            debug!("no more chunk on stream");
            if !buffer.trim().is_empty() {
                if let Ok(response) = serde_json::from_str::<OllamaResponse>(buffer.trim()) {
                    yield response;
                }
            }
        };
        Box::pin(fut)
    }
}
