import "leaflet/dist/leaflet.css";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Polygon,
  CircleMarker,
  Popup,
  useMapEvents,
} from "react-leaflet";
import { useEffect, useMemo, useState } from "react";
import StartScreen from "./components/StartScreen";
import HintsButton from "./components/HintsButton";

const API_URL = "https://bikeshare-analysis.onrender.com/predict";

const AUSTIN_CENTER = [30.2672, -97.7431];
const INITIAL_ZOOM = 13;

const OUTER_RING = [
  [90, -180],
  [90, 180],
  [-90, 180],
  [-90, -180],
];

const COUNTY_MASK_STEPS = [
  { scale: 1, opacity: 0.12 },
  { scale: 1.035, opacity: 0.16 },
  { scale: 1.08, opacity: 0.22 },
  { scale: 1.16, opacity: 0.3 },
  { scale: 1.28, opacity: 0.42 },
  { scale: 1.45, opacity: 0.58 },
];

// -----------------------------
// GeoJSON helpers
// -----------------------------
function getCountyOuterRings(geojson) {
  if (!geojson?.features?.length) {
    return [];
  }

  const geometry = geojson.features[0].geometry;

  if (geometry.type === "Polygon") {
    return [geometry.coordinates[0].map(([lng, lat]) => [lat, lng])];
  }

  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.map((polygon) =>
      polygon[0].map(([lng, lat]) => [lat, lng])
    );
  }

  return [];
}

function getRingsCenter(rings) {
  const points = rings.flat();

  if (!points.length) {
    return {
      lat: AUSTIN_CENTER[0],
      lng: AUSTIN_CENTER[1],
    };
  }

  const totals = points.reduce(
    (acc, [lat, lng]) => {
      acc.lat += lat;
      acc.lng += lng;
      return acc;
    },
    { lat: 0, lng: 0 }
  );

  return {
    lat: totals.lat / points.length,
    lng: totals.lng / points.length,
  };
}

function scaleCountyRings(rings, scale, center) {
  if (scale === 1) {
    return rings;
  }

  return rings.map((ring) =>
    ring.map(([lat, lng]) => [
      center.lat + (lat - center.lat) * scale,
      center.lng + (lng - center.lng) * scale,
    ])
  );
}

// -----------------------------
// Format helpers
// -----------------------------
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

// -----------------------------
// Map click marker
// -----------------------------
function ClickPoint({ selectedPoint, onPointSelect }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;

      onPointSelect({
        lat: Number(lat.toFixed(6)),
        lon: Number(lng.toFixed(6)),
        id: Date.now(),
      });
    },
  });

  if (!selectedPoint) return null;

  return (
    <>
      <CircleMarker
        key={`pulse-${selectedPoint.id}`}
        center={[selectedPoint.lat, selectedPoint.lon]}
        radius={14}
        pathOptions={{
          color: "#007bba",
          weight: 2,
          fillColor: "#007bba",
          fillOpacity: 0.15,
          className: "pulse-ring",
        }}
      />

      <CircleMarker
        center={[selectedPoint.lat, selectedPoint.lon]}
        radius={6}
        pathOptions={{
          color: "#007bba",
          weight: 2,
          fillColor: "#007bba",
          fillOpacity: 1,
        }}
      >
        <Popup>
          Candidate station
          <br />
          Lat: {selectedPoint.lat}
          <br />
          Lon: {selectedPoint.lon}
        </Popup>
      </CircleMarker>
    </>
  );
}

// -----------------------------
// Floating map controls
// -----------------------------
function ScoreButton({ selectedPoint, isLoading, onScore }) {
  return (
    <button
      onClick={onScore}
      disabled={!selectedPoint || isLoading}
      style={{
        position: "absolute",
        top: "20px",
        right: "20px",
        zIndex: 1100,
        padding: "12px 18px",
        backgroundColor: selectedPoint ? "#007bba" : "#9aa8af",
        color: "white",
        border: "none",
        borderRadius: "999px",
        cursor: selectedPoint && !isLoading ? "pointer" : "not-allowed",
        fontWeight: "bold",
        boxShadow: "0 4px 14px rgba(0,0,0,0.18)",
        transition: "transform 0.18s ease, background 0.18s ease",
      }}
    >
      {isLoading ? "Scoring..." : "Score Location"}
    </button>
  );
}

function SelectedPointCard({ selectedPoint }) {
  if (!selectedPoint) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: "76px",
        right: "20px",
        zIndex: 1100,
        background: "white",
        padding: "11px 14px",
        borderRadius: "12px",
        boxShadow: "0 4px 14px rgba(0,0,0,0.14)",
        fontSize: "13px",
        lineHeight: "1.5",
      }}
    >
      <div>
        <b>Lat:</b> {selectedPoint.lat}
      </div>
      <div>
        <b>Lon:</b> {selectedPoint.lon}
      </div>
    </div>
  );
}

// -----------------------------
// Results components
// -----------------------------
function StationComparisonSummary({ stationComparison }) {
  if (!stationComparison) return null;

  return (
    <div
      style={{
        marginBottom: "14px",
        padding: "10px",
        background: "#f4f7f9",
        borderRadius: "10px",
        borderLeft: "4px solid #007bba",
      }}
    >
      <div style={{ fontWeight: "bold", marginBottom: "4px" }}>
        Compared to Existing Stations
      </div>

      <div>
        Rank: {stationComparison.rank_position} of{" "}
        {stationComparison.total_stations_plus_candidate}
      </div>

      <div>
        Percentile: {formatDecimal(stationComparison.rank_percentile, 1)}%
      </div>
    </div>
  );
}

function StationRankingList({ stationComparison }) {
  const rows = stationComparison?.all_station_rankings;

  if (!rows?.length) return null;

  const maxTrips = Math.max(...rows.map((row) => row.trips_per_dock || 0));

  return (
    <div style={{ marginBottom: "16px" }}>
      <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
        Station Ranking
      </h4>

      <div
        style={{
          maxHeight: "260px",
          overflowY: "auto",
          paddingRight: "4px",
        }}
      >
        {rows.map((row) => {
          const widthPercent =
            maxTrips > 0
              ? Math.max(3, (row.trips_per_dock / maxTrips) * 100)
              : 3;

          return (
            <div
              key={`${row.rank}-${row.name}`}
              style={{
                marginBottom: "8px",
                padding: row.is_candidate ? "7px" : "0",
                borderRadius: "8px",
                background: row.is_candidate ? "#e6f4fa" : "transparent",
                border: row.is_candidate ? "1px solid #007bba" : "none",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "8px",
                  fontSize: "12px",
                  marginBottom: "3px",
                  fontWeight: row.is_candidate ? "bold" : "normal",
                }}
              >
                <span
                  style={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  #{row.rank} {row.is_candidate ? "Your location" : row.name}
                </span>

                <span>{formatNumber(row.trips_per_dock)}</span>
              </div>

              <div
                style={{
                  height: "8px",
                  background: "#e5e5e5",
                  borderRadius: "999px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${widthPercent}%`,
                    background: row.is_candidate ? "#007bba" : "#b8c4cc",
                    borderRadius: "999px",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function NearbyAttributes({ featureResults }) {
  if (!featureResults) return null;

  return (
    <div
      style={{
        marginBottom: "14px",
        padding: "10px",
        background: "#fafafa",
        borderRadius: "10px",
      }}
    >
      <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
        Nearby Attributes
      </h4>

      <div>
        <b>Transit Stops, 275m:</b> {featureResults.count_transit_stop_275m}
      </div>

      <div>
        <b>Amenities, 275m:</b> {featureResults.count_amenities_275m}
      </div>

      <div>
        <b>Jobs, 275m:</b> {featureResults.jobs_count_within_275m}
      </div>

      <div style={{ marginTop: "8px" }}>
        <b>Nearest Station:</b>{" "}
        {formatNumber(featureResults.nearest_bikeshare_station_m)} m
      </div>

      <div>
        <b>Population Density:</b>{" "}
        {formatDecimal(featureResults.population_density, 4)}
      </div>
    </div>
  );
}

function FeatureSummary({ topSummary }) {
  if (!topSummary?.length) return null;

  return (
    <div style={{ marginTop: "16px" }}>
      <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
        Why this score?
      </h4>

      {topSummary.slice(0, 5).map((row) => {
        const helpsPrediction = row.shap_value >= 0;

        return (
          <div
            key={row.feature}
            style={{
              marginBottom: "8px",
              padding: "8px",
              borderRadius: "8px",
              background: "#f4f7f9",
              borderLeft: `4px solid ${helpsPrediction ? "#007bba" : "#999"}`,
            }}
          >
            <div>
              <b>{row.feature}</b>
            </div>

            <div>{helpsPrediction ? "Increases" : "Decreases"} prediction</div>

            <div style={{ fontSize: "12px", color: "#555" }}>
              Percentile: {Math.round(row.percentile_rank)} ·{" "}
              {row.relative_to_median}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResultsPanel({
  featureResults,
  prediction,
  stationComparison,
  topSummary,
}) {
  if (!featureResults) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: "20px",
        left: "20px",
        zIndex: 1100,
        background: "white",
        padding: "16px",
        borderRadius: "14px",
        boxShadow: "0 6px 18px rgba(0,0,0,0.18)",
        width: "370px",
        maxHeight: "78vh",
        overflowY: "auto",
        fontSize: "14px",
        animation: "fadeUp 0.35s ease both",
      }}
    >
      <h3 style={{ marginBottom: "10px", color: "#007bba" }}>
        Location Insights
      </h3>

      {prediction !== null && prediction !== undefined && (
        <div
          style={{
            fontSize: "24px",
            fontWeight: "bold",
            color: "#007bba",
            marginBottom: "12px",
            lineHeight: "1.15",
          }}
        >
          Predicted Trips per Dock: {formatNumber(prediction)}
        </div>
      )}

      <StationComparisonSummary stationComparison={stationComparison} />
      <StationRankingList stationComparison={stationComparison} />
      <NearbyAttributes featureResults={featureResults} />
      <FeatureSummary topSummary={topSummary} />
    </div>
  );
}

// -----------------------------
// Main app
// -----------------------------
export default function App() {
  const shouldStartOnMap = window.location.hash === "#play";

  const [gameStarted, setGameStarted] = useState(shouldStartOnMap);
  const [showStartScreen, setShowStartScreen] = useState(!shouldStartOnMap);
  const [animateMapEntry, setAnimateMapEntry] = useState(false);

  const [countyGeoJson, setCountyGeoJson] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);

  const [featureResults, setFeatureResults] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [topSummary, setTopSummary] = useState([]);
  const [stationComparison, setStationComparison] = useState(null);

  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetch("/travis_county.geojson")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`GeoJSON load failed: ${res.status}`);
        }

        return res.json();
      })
      .then((data) => {
        console.log("Loaded Travis County GeoJSON:", data);
        setCountyGeoJson(data);
      })
      .catch((err) => {
        console.error("GeoJSON load error:", err);
      });
  }, []);

  useEffect(() => {
    function handlePopState() {
      if (window.location.hash !== "#play") {
        handleBackToStart();
      }
    }

    window.addEventListener("popstate", handlePopState);

    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  useEffect(() => {
    if (!animateMapEntry) return;

    const timeout = window.setTimeout(() => {
      setAnimateMapEntry(false);
    }, 3600);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [animateMapEntry]);

  const countyRings = useMemo(() => {
    return getCountyOuterRings(countyGeoJson);
  }, [countyGeoJson]);

  const countyMaskLayers = useMemo(() => {
    if (!countyRings.length) return [];

    const center = getRingsCenter(countyRings);

    return COUNTY_MASK_STEPS.map((step) => ({
      opacity: step.opacity,
      rings: scaleCountyRings(countyRings, step.scale, center),
    }));
  }, [countyRings]);

  function handlePlayStart() {
    if (window.location.hash !== "#play") {
      window.history.pushState({ screen: "map" }, "", "#play");
    }

    setGameStarted(true);
    setAnimateMapEntry(true);
  }

  function handleBackToStart() {
    setGameStarted(false);
    setShowStartScreen(true);
    setAnimateMapEntry(false);

    setSelectedPoint(null);
    setFeatureResults(null);
    setPrediction(null);
    setTopSummary([]);
    setStationComparison(null);
    setIsLoading(false);
  }

  function handlePointSelect(point) {
    setSelectedPoint(point);

    setFeatureResults(null);
    setPrediction(null);
    setTopSummary([]);
    setStationComparison(null);
  }

  async function getFeatures() {
    if (!selectedPoint) {
      alert("Click the map first!");
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          lat: selectedPoint.lat,
          lon: selectedPoint.lon,
        }),
      });

      const data = await response.json();

      console.log("Prediction response:", data);

      if (!data.success) {
        console.error("Backend returned error:", data.error);
        alert(`Backend error: ${data.error}`);
        return;
      }

      setFeatureResults(data.features);
      setPrediction(data.predicted_trips_per_dock);
      setTopSummary(data.top_feature_summary || []);
      setStationComparison(data.station_comparison);
    } catch (error) {
      console.error("Fetch failed:", error);
      alert("Could not reach backend. Check the browser console.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      {gameStarted && (
        <div className="map-screen">
          <div className="map-frame">
            <div
              className={`map-reveal-window ${
                animateMapEntry ? "is-entering" : ""
              }`}
            >
              <ScoreButton
                selectedPoint={selectedPoint}
                isLoading={isLoading}
                onScore={getFeatures}
              />

              <SelectedPointCard selectedPoint={selectedPoint} />

              <ResultsPanel
                featureResults={featureResults}
                prediction={prediction}
                stationComparison={stationComparison}
                topSummary={topSummary}
              />

              <MapContainer
                center={AUSTIN_CENTER}
                zoom={INITIAL_ZOOM}
                zoomControl={false}
                scrollWheelZoom={true}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                  url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                  subdomains={["a", "b", "c", "d"]}
                  maxZoom={20}
                />

                {countyMaskLayers.map((layer, index) => (
                  <Polygon
                    key={`county-mask-${index}`}
                    positions={[OUTER_RING, ...layer.rings]}
                    pathOptions={{
                      stroke: false,
                      fillColor: "#ffffff",
                      fillOpacity: layer.opacity,
                      fillRule: "evenodd",
                      interactive: false,
                    }}
                  />
                ))}

                {countyGeoJson && (
                  <GeoJSON
                    data={countyGeoJson}
                    style={() => ({
                      color: "#007bba",
                      weight: 2.5,
                      opacity: 0.95,
                      dashArray: "8 6",
                      lineCap: "round",
                      lineJoin: "round",
                      fill: false,
                    })}
                  />
                )}

                <ClickPoint
                  selectedPoint={selectedPoint}
                  onPointSelect={handlePointSelect}
                />
              </MapContainer>
            </div>
          </div>

          <HintsButton className="map-screen-hints" />
        </div>
      )}

      {showStartScreen && (
        <StartScreen
          onPlayStart={handlePlayStart}
          onTransitionComplete={() => setShowStartScreen(false)}
        />
      )}
    </div>
  );
}