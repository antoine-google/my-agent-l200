# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  service_account_name = "${var.service_name}-sa"
  storage_bucket_name  = "${var.service_name}-storage-${random_id.suffix.hex}"
  repo_name            = "${var.service_name}-repo"
  secret_id            = "gemini-api-key"
}

# ---------------------------------------------------------------------------
# 1. Google Cloud APIs Enablement
# ---------------------------------------------------------------------------
locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "aiplatform.googleapis.com",
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(locals.required_services)
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# 2. Least-Privilege Runtime Service Account
# ---------------------------------------------------------------------------
resource "google_service_account" "app_sa" {
  account_id   = locals.service_account_name
  display_name = "Runtime Service Account for ${var.service_name}"
  project      = var.project_id

  depends_on = [google_project_service.enabled_apis]
}

locals {
  app_sa_roles = [
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectUser",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/aiplatform.user",
  ]
}

resource "google_project_iam_member" "app_sa_permissions" {
  for_each = toset(locals.app_sa_roles)
  project  = var.project_id
  role     = each.key
  member   = "serviceAccount:${google_service_account.app_sa.email}"
}

# ---------------------------------------------------------------------------
# 3. Artifact Registry (Container Images)
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "agent_repo" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = locals.repo_name
  description   = "Docker container repository for ${var.service_name}"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_apis]
}

# ---------------------------------------------------------------------------
# 4. Secret Manager (Gemini API Key)
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = locals.secret_id
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

# ---------------------------------------------------------------------------
# 5. Cloud Storage Bucket (Vector Memory Backups & Agent Artifacts)
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "agent_storage" {
  name                        = locals.storage_bucket_name
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
  }

  lifecycle_rule {
    action {
      type = "AbortIncompleteMultipartUpload"
    }
    condition {
      age = 1
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# Grant Cloud Run service account access to its dedicated storage bucket
resource "google_storage_bucket_iam_member" "agent_storage_access" {
  bucket = google_storage_bucket.agent_storage.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.app_sa.email}"
}

# ---------------------------------------------------------------------------
# 6. Cloud Run Service (Agent Runtime Container)
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "agent_service" {
  name     = var.service_name
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.app_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    max_instance_request_concurrency = var.concurrency

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}/${var.service_name}:${var.image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      # Strategic Model Routing & Runtime Configuration
      env {
        name  = "COORDINATOR_MODEL"
        value = var.coordinator_model
      }
      env {
        name  = "NEWS_MODEL"
        value = var.news_model
      }
      env {
        name  = "METRICS_MODEL"
        value = var.metrics_model
      }
      env {
        name  = "SUMMARIZER_MODEL"
        value = var.summarizer_model
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "false"
      }
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "STORAGE_BUCKET_NAME"
        value = google_storage_bucket.agent_storage.name
      }

      # Secret Key Mount from Secret Manager
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      # Startup Probe
      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 3
        tcp_socket {
          port = 8080
        }
      }

      # Liveness Probe
      liveness_probe {
        initial_delay_seconds = 15
        timeout_seconds       = 3
        period_seconds        = 15
        failure_threshold     = 3
        tcp_socket {
          port = 8080
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_project_iam_member.app_sa_permissions,
    google_storage_bucket_iam_member.agent_storage_access,
  ]
}

# Optional Unauthenticated Access
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count    = var.enable_public_access ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.agent_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# 7. Cloud Logging Sink for Intent-vs-Outcome Auditing
# ---------------------------------------------------------------------------
resource "google_logging_project_sink" "audit_log_sink" {
  name                   = "${var.service_name}-intent-outcome-sink"
  project                = var.project_id
  destination            = "storage.googleapis.com/${google_storage_bucket.agent_storage.name}"
  filter                 = "resource.type=\"cloud_run_revision\" AND jsonPayload.event_type=\"intent_vs_outcome\""
  unique_writer_identity = true

  depends_on = [google_storage_bucket.agent_storage]
}

resource "google_storage_bucket_iam_member" "log_sink_storage_writer" {
  bucket = google_storage_bucket.agent_storage.name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.audit_log_sink.writer_identity
}
