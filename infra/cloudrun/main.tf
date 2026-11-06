# Isolated validation runner as a Cloud Run Job (S4-VAL-14).
#
# STATUS: NOT DEPLOYED. Deployment is blocked on GitHub Actions Workload
# Identity Federation (prerequisite S0-INF-06.4) and on a GCP project owned by
# the release authority. This definition is the reviewed, pinned target so
# the job can be applied once those exist.
#
# Isolation: the job runs a digest-pinned image built from this repository's
# Dockerfile (requirements.lock, digest-pinned base), with no public ingress,
# a dedicated least-privilege service account, egress limited to Google APIs
# via the VPC connector, and results written only to a versioned bucket.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "europe-west1"
}
variable "image" {
  description = "Image reference pinned by digest, e.g. REGION-docker.pkg.dev/PROJECT/apgi/validation@sha256:..."
  type        = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image))
    error_message = "The validation image must be pinned by digest (APGI-IMP-001)."
  }
}
variable "results_bucket" { type = string }
variable "vpc_connector" {
  type    = string
  default = null
}

resource "google_service_account" "runner" {
  project      = var.project_id
  account_id   = "apgi-validation-runner"
  display_name = "APGI isolated validation runner"
}

resource "google_storage_bucket_iam_member" "results_writer" {
  bucket = var.results_bucket
  # Cloud Storage FUSE needs create + read of the objects it writes.
  role = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_cloud_run_v2_job" "validation" {
  project  = var.project_id
  name     = "apgi-validation-runner"
  location = var.region

  template {
    task_count  = 1
    parallelism = 1
    template {
      service_account = google_service_account.runner.email
      max_retries     = 0
      timeout         = "7200s"
      containers {
        image   = var.image
        command = ["python", "main.py"]
        # /app/results is created in the image and owned by the non-root
        # `apgi` user; it is backed by the results bucket (volume below), so
        # reports persist beyond the ephemeral container.
        args = ["validate", "--all-protocols", "--output-dir", "/app/results"]
        env {
          name  = "APGI_ENV"
          value = "production"
        }
        env {
          name  = "APGI_MCMC_CORES"
          value = "1"
        }
        env {
          name = "APGI_MASTER_KEY"
          value_source {
            secret_key_ref {
              secret  = "apgi-master-key"
              version = "latest"
            }
          }
        }
        resources {
          limits = { cpu = "4", memory = "16Gi" }
        }
        volume_mounts {
          name       = "results"
          mount_path = "/app/results"
        }
      }
      volumes {
        name = "results"
        gcs {
          bucket    = var.results_bucket
          read_only = false
        }
      }
      dynamic "vpc_access" {
        for_each = var.vpc_connector == null ? [] : [1]
        content {
          connector = var.vpc_connector
          egress    = "ALL_TRAFFIC"
        }
      }
    }
  }
}

output "job_name" { value = google_cloud_run_v2_job.validation.name }
