import type {
  HealthResponse,
  FeaturesResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "@/lib/types";

async function fetchApi<T>(endpoint: string): Promise<T> {
  const response = await fetch(`/api/stocksense/${endpoint}`, {
    method: "GET",
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(
      `API request failed (${response.status}): ${endpoint}`
    );
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchApi<HealthResponse>("health");
}

export async function getPredictions(): Promise<PredictionResponse> {
  return fetchApi<PredictionResponse>("predict/latest");
}

export async function getFeatures(): Promise<FeaturesResponse> {
  return fetchApi<FeaturesResponse>("features/latest");
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return fetchApi<ModelInfoResponse>("model-info");
}