#!/usr/bin/env bash
# One-command deploy of ShelfTalk to Cloud Run (run in Google Cloud Shell).
#
#   Option A (recommended, no key needed): Gemini via Vertex AI
#       ./deploy.sh
#   Option B: Gemini API key from aistudio.google.com
#       GEMINI_API_KEY=xxxx ./deploy.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-shelftalk}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "No project set. Run: gcloud config set project YOUR_PROJECT_ID"; exit 1
fi
echo "Deploying $SERVICE to $PROJECT_ID ($REGION)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com firestore.googleapis.com aiplatform.googleapis.com \
  --project "$PROJECT_ID"

# Firestore database (skips if it already exists)
gcloud firestore databases create --project "$PROJECT_ID" --location="$REGION" \
  --type=firestore-native >/dev/null 2>&1 || true

# Let the Cloud Run service account use Firestore and Vertex AI
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for ROLE in roles/datastore.user roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" \
    --role="$ROLE" --condition=None >/dev/null
done

ENV_VARS="STORE=firestore,SHOP_ID=demo-shop"
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  ENV_VARS="$ENV_VARS,GEMINI_API_KEY=$GEMINI_API_KEY"
else
  ENV_VARS="$ENV_VARS,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=global"
fi

gcloud run deploy "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --source . \
  --allow-unauthenticated --memory 1Gi --cpu 1 --timeout 300 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars "$ENV_VARS"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
echo
echo "ShelfTalk is live: $URL"
