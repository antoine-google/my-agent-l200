variable "project_id" {
  description = "The Google Cloud Project ID where resources will be provisioned."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run, Artifact Registry, and Cloud Storage."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service and application prefix."
  type        = string
  default     = "alphabet-agent-l200"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, staging, prod)."
  type        = string
  default     = "production"
}

variable "image_tag" {
  description = "Container image tag to deploy on Cloud Run."
  type        = string
  default     = "latest"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances (set to 1 to eliminate cold starts)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances."
  type        = number
  default     = 5
}

variable "concurrency" {
  description = "Maximum number of concurrent requests per Cloud Run container instance."
  type        = number
  default     = 16
}

variable "cpu" {
  description = "CPU limit for the Cloud Run container."
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory limit for the Cloud Run container."
  type        = string
  default     = "4Gi"
}

variable "enable_public_access" {
  description = "Allow unauthenticated invocations on Cloud Run. Keep false for private/IAP-protected setups."
  type        = bool
  default     = false
}

variable "coordinator_model" {
  description = "Gemini model identifier for Root Coordinator Agent (high-reasoning, synthesis)."
  type        = string
  default     = "gemini-3.1-pro-preview"
}

variable "news_model" {
  description = "Gemini model identifier for News & Sentiment Specialist Agent (multimodal, fast)."
  type        = string
  default     = "gemini-3.8-flash"
}

variable "metrics_model" {
  description = "Gemini model identifier for Market Metrics Specialist Agent (ultra-low latency)."
  type        = string
  default     = "gemini-3.5-flash-lite"
}

variable "summarizer_model" {
  description = "Gemini model identifier for Context Compaction Summarizer."
  type        = string
  default     = "gemini-3.5-flash-lite"
}
