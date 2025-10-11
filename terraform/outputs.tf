output "api_url" {
  description = "URL of the RAG API service"
  value       = google_cloud_run_service.api.status[0].url
}

output "ui_url" {
  description = "URL of the RAG UI service"
  value       = google_cloud_run_service.ui.status[0].url
}

output "etl_url" {
  description = "URL of the ETL service"
  value       = google_cloud_run_service.etl.status[0].url
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.rag_service.email
}

output "artifact_registry" {
  description = "Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "deployment_commands" {
  description = "Commands to deploy services"
  value       = <<-EOT
  # Build and push images:
  ./deploy.sh build
  
  # Deploy to Cloud Run:
  ./deploy.sh deploy
  
  # Access services:
  UI:  ${google_cloud_run_service.ui.status[0].url}
  API: ${google_cloud_run_service.api.status[0].url}
  EOT
}

