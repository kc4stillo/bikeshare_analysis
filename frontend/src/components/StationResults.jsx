import React from "react";
import "./StationResults.css";

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  return Math.round(Number(value)).toLocaleString();
}

function formatDecimal(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  return Number(value).toFixed(digits);
}

function formatFeatureName(feature) {
  return String(feature || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function ImpactList({ title, rows, type }) {
  if (!rows || rows.length === 0) return null;

  return (
    <div className="impact-section">
      <div className="results-section-header">{title}</div>

      <div className="impact-list">
        {rows.map((row) => (
          <div className="impact-row" key={`${type}-${row.feature}`}>
            <div>
              <div className="impact-name">{formatFeatureName(row.feature)}</div>
              <div className="impact-meta">
                Percentile {Math.round(row.percentile_rank || 0)} ·{" "}
                {row.relative_to_median || "relative to median"}
              </div>
            </div>

            <div className={`impact-pill ${type}`}>
              {type === "positive" ? "Helps" : "Hurts"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ScoringOverlay() {
  return (
    <div className="scoring-overlay">
      <div className="scoring-card">
        <div className="scoring-orbit">
          <span />
          <span />
          <span />
        </div>

        <div>
          <h2>Analyzing your station...</h2>
          <p>
            Reading nearby population, jobs, transit, amenities, and existing
            station context.
          </p>
        </div>

        <div className="scoring-progress">
          <span />
        </div>
      </div>
    </div>
  );
}

export default function StationResults({
  featureResults,
  prediction,
  stationComparison,
  topSummary = [],
  isCollapsed,
  onExpand,
  onTryAnother,
}) {
  if (!featureResults) return null;

  if (isCollapsed) {
    return (
      <div className="results-compact-actions">
        <button
          className="compact-results-button primary"
          type="button"
          onClick={onExpand}
        >
          Station Stats
        </button>

        <button
          className="compact-results-button secondary"
          type="button"
          onClick={onTryAnother}
        >
          Play Again
        </button>
      </div>
    );
  }

  const positiveFactors = topSummary
    .filter((row) => Number(row.shap_value) >= 0)
    .slice(0, 3);

  const negativeFactors = topSummary
    .filter((row) => Number(row.shap_value) < 0)
    .slice(0, 3);

  const rankings = stationComparison?.all_station_rankings || [];
  const maxTrips = Math.max(
    ...rankings.map((row) => Number(row.trips_per_dock) || 0),
    1
  );

  const percentile = Number(stationComparison?.rank_percentile) || 0;

  return (
    <aside className="station-results-panel">
      <div className="results-eyebrow">Station Score</div>

      <div className="score-hero">
        <div>
          <div className="score-number">{formatNumber(prediction)}</div>
          <div className="score-label">predicted trips per dock</div>
        </div>

        {stationComparison && (
          <div className="score-rank-card">
            <span>Rank</span>
            <strong>
              {stationComparison.rank_position} /{" "}
              {stationComparison.total_stations_plus_candidate}
            </strong>
          </div>
        )}
      </div>

      {stationComparison && (
        <div className="percentile-card">
          <div>
            <span>Percentile</span>
            <strong>{formatDecimal(percentile, 1)}%</strong>
          </div>

          <div className="percentile-track">
            <div
              className="percentile-fill"
              style={{
                width: `${Math.min(100, Math.max(0, percentile))}%`,
              }}
            />
          </div>
        </div>
      )}

      <div className="snapshot-grid">
        <div className="snapshot-card">
          <strong>{formatNumber(featureResults.count_transit_stop_275m)}</strong>
          <span>Transit stops nearby</span>
        </div>

        <div className="snapshot-card">
          <strong>{formatNumber(featureResults.count_amenities_275m)}</strong>
          <span>Amenities nearby</span>
        </div>

        <div className="snapshot-card">
          <strong>{formatNumber(featureResults.jobs_count_within_275m)}</strong>
          <span>Jobs nearby</span>
        </div>

        <div className="snapshot-card">
          <strong>
            {formatNumber(featureResults.nearest_bikeshare_station_m)}m
          </strong>
          <span>Nearest station</span>
        </div>
      </div>

      <ImpactList title="What helped" rows={positiveFactors} type="positive" />
      <ImpactList title="What hurt" rows={negativeFactors} type="negative" />

      {rankings.length > 0 && (
        <div className="leaderboard-section">
          <div className="results-section-header">Station Leaderboard</div>

          <div className="leaderboard-list">
            {rankings.slice(0, 12).map((row) => {
              const trips = Number(row.trips_per_dock) || 0;
              const widthPercent = Math.max(4, (trips / maxTrips) * 100);

              return (
                <div
                  className={`leaderboard-row ${
                    row.is_candidate ? "is-candidate" : ""
                  }`}
                  key={`${row.rank}-${row.name}`}
                >
                  <div className="leaderboard-topline">
                    <span>
                      #{row.rank} {row.is_candidate ? "Your station" : row.name}
                    </span>
                    <strong>{formatNumber(trips)}</strong>
                  </div>

                  <div className="leaderboard-bar">
                    <div style={{ width: `${widthPercent}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <button className="try-another-button" onClick={onTryAnother} type="button">
        Try Another Location
      </button>
    </aside>
  );
}