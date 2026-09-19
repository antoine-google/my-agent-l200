output "cloud_run_service_url" {
  description = "The HTTPS invocation URL of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.agent_service.uri
}

output "artifact_registry_repository_url" {
  description = "The Docker repository URL in Google Artifact Registry."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}"
}

output "service_account_email" {
  description = "The email of the dedicated least-privilege runtime service account."
  value       = google_service_account.app_sa.email
}

output "storage_bucket_name" {
  description = "The Cloud Storage bucket provisioned for agent memory backups and artifacts."
  value       = google_storage_bucket.agent_storage.name
}

output "secret_manager_secret_id" {
  description = "The Google Secret Manager secret resource ID for the Gemini API key."
  value       = google_secret_manager_secret.gemini_api_key.secret_id
}
