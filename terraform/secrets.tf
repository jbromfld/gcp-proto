# Secret versions (populated via terraform or manually)
resource "google_secret_manager_secret_version" "elasticsearch_url" {
  secret = google_secret_manager_secret.elasticsearch_url.id

  secret_data = var.elasticsearch_url

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_version" "elasticsearch_password" {
  secret = google_secret_manager_secret.elasticsearch_password.id

  secret_data = var.elasticsearch_password

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Grant service account access to secrets
resource "google_secret_manager_secret_iam_member" "elasticsearch_url_access" {
  secret_id = google_secret_manager_secret.elasticsearch_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_service.email}"
}

resource "google_secret_manager_secret_iam_member" "elasticsearch_password_access" {
  secret_id = google_secret_manager_secret.elasticsearch_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_service.email}"
}

