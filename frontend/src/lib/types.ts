export interface ClassificationPrediction {
  predicted_direction: "Up" | "Down";
  probability_up: number;
  probability_down: number;
}

export interface RegressionPrediction {
  predicted_return: number;
  predicted_return_percent: number;
}

export interface PredictionResponse {
  symbol: string;
  market_data_date: string;
  latest_close: number;
  forecast_horizon: string;
  classification: Record<string, ClassificationPrediction>;
  regression: Record<string, RegressionPrediction>;
  disclaimer: string;
}

export interface FeaturesResponse {
  date: string;
  close: number;
  features: Record<string, number>;
}

export interface ModelInfoResponse {
  project: string;
  symbol: string;
  forecast_horizon: string;
  features_count: number;
  classifiers: string[];
  regressors: string[];
  model_status: string;
  prediction_status: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  models_loaded: {
    classifiers: string[];
    regressors: string[];
  };
  timestamp: string;
}