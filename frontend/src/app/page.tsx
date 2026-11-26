"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Clock3,
  Database,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
  Wifi,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getFeatures,
  getHealth,
  getModelInfo,
  getPredictions,
} from "@/lib/api";

import type {
  FeaturesResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "@/lib/types";

const DISCLAIMER =
  "Experimental machine learning predictions only. Models have not demonstrated reliable predictive performance on the held-out test period. Not investment advice.";

const panel =
  "rounded-xl border border-slate-800 bg-[#111827] shadow-lg shadow-black/10";

function formatNumber(value: number, digits = 2) {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatFeatureName(value: string) {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Activity;
}) {
  return (
    <div className={`${panel} p-5`}>
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">{label}</p>
        <Icon className="h-5 w-5 text-cyan-400" />
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-white">
        {value}
      </p>
      <p className="mt-2 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

function SectionTitle({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-5">
      <h2 className="text-lg font-semibold text-white">{title}</h2>
      <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
    </div>
  );
}

function StatusDot({ healthy }: { healthy: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${
        healthy ? "bg-emerald-400" : "bg-rose-400"
      }`}
    />
  );
}

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [predictions, setPredictions] =
    useState<PredictionResponse | null>(null);
  const [features, setFeatures] = useState<FeaturesResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [showFeatures, setShowFeatures] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadDashboard = useCallback(async (manual = false) => {
    if (manual) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    setError("");

    try {
      const [healthData, predictionData, featureData, modelData] =
        await Promise.all([
          getHealth(),
          getPredictions(),
          getFeatures(),
          getModelInfo(),
        ]);

      setHealth(healthData);
      setPredictions(predictionData);
      setFeatures(featureData);
      setModelInfo(modelData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load StockSense data.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    async function loadInitialDashboard() {
      try {
        const [healthData, predictionData, featureData, modelData] =
          await Promise.all([
            getHealth(),
            getPredictions(),
            getFeatures(),
            getModelInfo(),
          ]);

        if (!active) return;

        setHealth(healthData);
        setPredictions(predictionData);
        setFeatures(featureData);
        setModelInfo(modelData);
        setLastUpdated(new Date());
      } catch (err) {
        if (!active) return;

        setError(
          err instanceof Error
            ? err.message
            : "Unable to load StockSense data.",
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadInitialDashboard();

    return () => {
      active = false;
    };
  }, []);

  const classificationData = useMemo(() => {
    if (!predictions) return [];

    return Object.entries(predictions.classification).map(
      ([model, result]) => ({
        model,
        Up: Number((result.probability_up * 100).toFixed(2)),
        Down: Number((result.probability_down * 100).toFixed(2)),
      }),
    );
  }, [predictions]);

  const regressionData = useMemo(() => {
    if (!predictions) return [];

    return Object.entries(predictions.regression).map(
      ([model, result]) => ({
        model,
        "Predicted return (%)": Number(
          result.predicted_return_percent.toFixed(4),
        ),
      }),
    );
  }, [predictions]);

  const consensus = useMemo(() => {
    if (!predictions) return null;

    const results = Object.values(predictions.classification);
    const upVotes = results.filter(
      (item) => item.predicted_direction === "Up",
    ).length;

    return {
      upVotes,
      downVotes: results.length - upVotes,
      total: results.length,
      percentage: results.length
        ? (Math.max(upVotes, results.length - upVotes) / results.length) * 100
        : 0,
    };
  }, [predictions]);

  const featureEntries = useMemo(
    () => Object.entries(features?.features ?? {}),
    [features],
  );

  const isHealthy = health?.status?.toLowerCase() === "healthy";

  return (
    <main className="min-h-screen bg-[#090e18] text-slate-200">
      <div className="mx-auto max-w-[1500px] px-4 py-5 sm:px-6 lg:px-8">
        <header className="mb-8 flex flex-col gap-5 border-b border-slate-800 pb-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-400/10 ring-1 ring-cyan-400/20">
              <BrainCircuit className="h-6 w-6 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white sm:text-2xl">
                StockSense <span className="text-cyan-400">AI</span>
              </h1>
              <p className="text-xs text-slate-500">
                Experimental NIFTY 50 forecasting & analytics
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs">
              <StatusDot healthy={Boolean(isHealthy)} />
              <span className="text-slate-300">
                {isHealthy ? "API Operational" : "API Unavailable"}
              </span>
            </div>
            <button
              onClick={() => void loadDashboard(true)}
              disabled={loading || refreshing}
              className="flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw
                className={`h-4 w-4 ${
                  refreshing ? "animate-spin" : ""
                }`}
              />
              Refresh
            </button>
          </div>
        </header>

        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
            <CircleAlert className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="font-semibold">Dashboard data unavailable</p>
              <p className="mt-1 text-rose-200/80">{error}</p>
              <button
                onClick={() => void loadDashboard(true)}
                className="mt-2 underline underline-offset-4"
              >
                Retry connection
              </button>
            </div>
          </div>
        )}

        {loading && !predictions ? (
          <div className="flex min-h-[420px] flex-col items-center justify-center gap-4">
            <RefreshCw className="h-8 w-8 animate-spin text-cyan-400" />
            <p className="text-sm text-slate-400">
              Connecting to StockSense API...
            </p>
          </div>
        ) : predictions && features && modelInfo ? (
          <>
            <div className="mb-7 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="NIFTY 50"
                value={formatNumber(predictions.latest_close)}
                detail={`Market data date: ${predictions.market_data_date}`}
                icon={TrendingUp}
              />
              <MetricCard
                label="Classification Models"
                value={String(modelInfo.classifiers.length)}
                detail="Logistic Regression · Random Forest · XGBoost"
                icon={BrainCircuit}
              />
              <MetricCard
                label="Regression Models"
                value={String(modelInfo.regressors.length)}
                detail="Ridge · Random Forest · XGBoost"
                icon={BarChart3}
              />
              <MetricCard
                label="Technical Features"
                value={String(modelInfo.features_count)}
                detail="Engineered model input features"
                icon={Activity}
              />
            </div>

            <section className={`${panel} mb-7 p-5 sm:p-6`}>
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                <div>
                  <p className="text-sm text-slate-400">Forecast horizon</p>
                  <h2 className="mt-1 text-xl font-semibold text-white">
                    {predictions.forecast_horizon}
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Latest available market data · {predictions.market_data_date}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-emerald-300">
                    {modelInfo.model_status}
                  </span>
                  <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-amber-300">
                    {modelInfo.prediction_status}
                  </span>
                </div>
              </div>
            </section>

            <div className="mb-7 grid grid-cols-1 gap-6 xl:grid-cols-3">
              <section className={`${panel} p-5 sm:p-6 xl:col-span-2`}>
                <SectionTitle
                  title="Classification probabilities"
                  subtitle="Predicted probability of upward and downward market direction"
                />
                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={classificationData}
                      margin={{ top: 10, right: 10, left: -15, bottom: 10 }}
                    >
                      <CartesianGrid
                        stroke="#263244"
                        strokeDasharray="3 3"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="model"
                        tick={{ fill: "#94a3b8", fontSize: 11 }}
                        axisLine={{ stroke: "#334155" }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: "#94a3b8", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value: number) => `${value}%`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#111827",
                          border: "1px solid #334155",
                          borderRadius: 10,
                          color: "#fff",
                        }}
                        formatter={(value) => [`${value}%`]}
                      />
                      <Legend />
                      <Bar
                        dataKey="Up"
                        fill="#10b981"
                        radius={[5, 5, 0, 0]}
                      />
                      <Bar
                        dataKey="Down"
                        fill="#f97373"
                        radius={[5, 5, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className={`${panel} p-5 sm:p-6`}>
                <SectionTitle
                  title="Model consensus"
                  subtitle="Agreement among classification models"
                />
                {consensus && (
                  <>
                    <div className="mb-5 flex items-end justify-between">
                      <div>
                        <p className="text-4xl font-bold text-white">
                          {consensus.upVotes}/{consensus.total}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          Models predicting Up
                        </p>
                      </div>
                      <div className="rounded-full bg-emerald-500/10 p-3">
                        <TrendingUp className="h-6 w-6 text-emerald-400" />
                      </div>
                    </div>
                    <div className="mb-3 h-3 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-emerald-400"
                        style={{
                          width: `${
                            consensus.total
                              ? (consensus.upVotes / consensus.total) * 100
                              : 0
                          }%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>{consensus.upVotes} Up</span>
                      <span>{consensus.downVotes} Down</span>
                    </div>
                    <div className="mt-6 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                      <p className="text-xs leading-5 text-amber-200/80">
                        Model agreement is not a measure of accuracy, certainty,
                        or future market performance.
                      </p>
                    </div>
                  </>
                )}
              </section>
            </div>

            <section className={`${panel} mb-7 p-5 sm:p-6`}>
              <SectionTitle
                title="Regression analysis"
                subtitle="Predicted next-session returns from the regression models"
              />
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={regressionData}
                    margin={{ top: 10, right: 15, left: 5, bottom: 10 }}
                  >
                    <CartesianGrid
                      stroke="#263244"
                      strokeDasharray="3 3"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="model"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      axisLine={{ stroke: "#334155" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value: number) => `${value}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#111827",
                        border: "1px solid #334155",
                        borderRadius: 10,
                      }}
                      formatter={(value) => [`${value}%`, "Predicted return"]}
                    />
                    <Bar
                      dataKey="Predicted return (%)"
                      fill="#38bdf8"
                      radius={[5, 5, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {Object.entries(predictions.regression).map(([name, result]) => {
                  const positive = result.predicted_return_percent >= 0;
                  return (
                    <div
                      key={name}
                      className="rounded-lg border border-slate-800 bg-slate-900/60 p-4"
                    >
                      <p className="text-xs text-slate-400">{name}</p>
                      <p
                        className={`mt-2 flex items-center gap-1 text-xl font-semibold ${
                          positive ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {positive ? (
                          <ArrowUpRight className="h-5 w-5" />
                        ) : (
                          <ArrowDownRight className="h-5 w-5" />
                        )}
                        {positive ? "+" : ""}
                        {formatNumber(result.predicted_return_percent, 4)}%
                      </p>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className={`${panel} mb-7 overflow-hidden`}>
              <button
                onClick={() => setShowFeatures((current) => !current)}
                className="flex w-full items-center justify-between p-5 text-left sm:p-6"
                aria-expanded={showFeatures}
              >
                <div>
                  <h2 className="font-semibold text-white">
                    Latest technical features
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">
                    {featureEntries.length} engineered indicators · Data date:{" "}
                    {features.date}
                  </p>
                </div>
                {showFeatures ? (
                  <ChevronUp className="h-5 w-5 text-slate-400" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-slate-400" />
                )}
              </button>
              {showFeatures && (
                <div className="overflow-x-auto border-t border-slate-800">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-900/70 text-xs uppercase text-slate-500">
                      <tr>
                        <th className="px-5 py-3 font-medium">Feature</th>
                        <th className="px-5 py-3 text-right font-medium">
                          Value
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {featureEntries.map(([name, value]) => (
                        <tr
                          key={name}
                          className="border-t border-slate-800/70 hover:bg-slate-900/50"
                        >
                          <td className="px-5 py-3 text-slate-300">
                            {formatFeatureName(name)}
                          </td>
                          <td className="px-5 py-3 text-right font-mono text-cyan-300">
                            {formatNumber(value, 6)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <section className={`${panel} mb-7 p-5 sm:p-6`}>
              <SectionTitle
                title="System information"
                subtitle="Backend connectivity and model inventory"
              />
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {[
                  {
                    label: "Backend API",
                    value: health?.service ?? "Unavailable",
                    icon: Wifi,
                  },
                  {
                    label: "Model status",
                    value: modelInfo.model_status,
                    icon: ShieldCheck,
                  },
                  {
                    label: "Prediction status",
                    value: modelInfo.prediction_status,
                    icon: BrainCircuit,
                  },
                  {
                    label: "Data source",
                    value: "Historical market data",
                    icon: Database,
                  },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="rounded-lg border border-slate-800 bg-slate-900/60 p-4"
                  >
                    <item.icon className="mb-3 h-5 w-5 text-cyan-400" />
                    <p className="text-xs text-slate-500">{item.label}</p>
                    <p className="mt-1 text-sm font-medium text-slate-200">
                      {item.value}
                    </p>
                  </div>
                ))}
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-slate-800 pt-4 text-xs text-slate-500">
                <span className="flex items-center gap-2">
                  <Clock3 className="h-3.5 w-3.5" />
                  {lastUpdated
                    ? `Dashboard refreshed at ${lastUpdated.toLocaleTimeString()}`
                    : "Waiting for data"}
                </span>
                <span className="flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5" />
                  Historical data · Not a live market feed
                </span>
              </div>
            </section>

            <section className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
              <div className="flex items-start gap-3">
                <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
                <div>
                  <h3 className="font-semibold text-amber-200">
                    Research disclaimer
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-amber-100/70">
                    {predictions.disclaimer || DISCLAIMER}
                  </p>
                  <p className="mt-2 text-xs leading-5 text-amber-100/50">
                    This dashboard is an experimental machine learning
                    demonstration. Model outputs are not reliable trading
                    signals, financial recommendations, or guarantees of
                    future market movement.
                  </p>
                </div>
              </div>
            </section>
          </>
        ) : (
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-8 text-center">
            <CircleAlert className="mx-auto h-8 w-8 text-amber-400" />
            <h2 className="mt-3 font-semibold text-white">
              Dashboard data is not available
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Check the backend connection and try again.
            </p>
            <button
              onClick={() => void loadDashboard(true)}
              className="mt-5 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950"
            >
              Retry
            </button>
          </div>
        )}

        <footer className="mt-8 border-t border-slate-800 py-5 text-center text-xs text-slate-600">
          StockSense AI · Experimental NIFTY 50 Forecasting & Analytics
        </footer>
      </div>
    </main>
  );
}