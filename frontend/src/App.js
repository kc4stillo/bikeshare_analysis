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

const AUSTIN_CENTER = [30.2672, -97.7431];

// giant outer rectangle covering the whole map world
const OUTER_RING = [
  [90, -180],
  [90, 180],
  [-90, 180],
  [-90, -180],
];

// convert GeoJSON geometry into Leaflet-style hole rings
function getCountyOuterRings(geojson) {
  if (!geojson || !geojson.features || geojson.features.length === 0) return [];

  const geometry = geojson.features[0].geometry;

  if (geometry.type === "Polygon") {
    // only outer ring
    return [geometry.coordinates[0].map(([lng, lat]) => [lat, lng])];
  }

  if (geometry.type === "MultiPolygon") {
    // each polygon -> take its outer ring
    return geometry.coordinates.map((polygon) =>
      polygon[0].map(([lng, lat]) => [lat, lng])
    );
  }

  return [];
}

function ClickPoint({ selectedPoint, setSelectedPoint }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      setSelectedPoint({
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

async function getFeatures() {
  if (!selectedPoint) return;

  const response = await fetch("https://bikeshare-analysis.onrender.com/predict", {
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

  if (data.success) {
    setFeatureResults(data.features);
    setPrediction(data.predicted_trips_per_dock);
    setTopSummary(data.top_feature_summary || []);
  }
}

export default function App() {
  const [countyGeoJson, setCountyGeoJson] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [featureResults, setFeatureResults] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [topSummary, setTopSummary] = useState([]);
  

  // load Travis County boundary
  useEffect(() => {
    fetch("/travis_county.geojson")
      .then((res) => res.json())
      .then((data) => setCountyGeoJson(data))
      .catch((err) => console.error("GeoJSON load error:", err));
  }, []);

  const countyHoles = useMemo(() => {
    return getCountyOuterRings(countyGeoJson);
  }, [countyGeoJson]);

  // 🔥 call backend
async function getFeatures() {
  if (!selectedPoint) return;

  const response = await fetch("https://bikeshare-analysis.onrender.com/predict", {
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

  if (data.success) {
    setFeatureResults(data.features);
    setPrediction(data.predicted_trips_per_dock);
    setTopSummary(data.top_feature_summary || []);
  }
}

  return (
    <div style={{ height: "100vh", width: "100%" }}>
      
      {/* BUTTON */}
      <button
        onClick={getFeatures}
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          zIndex: 1000,
          padding: "10px 14px",
          backgroundColor: "#007bba",
          color: "white",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        Get Features
      </button>

{featureResults && (
  <div
    style={{
      position: "absolute",
      bottom: "20px",
      left: "20px",
      zIndex: 1000,
      background: "white",
      padding: "16px",
      borderRadius: "10px",
      boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
      maxWidth: "340px",
      maxHeight: "75vh",
      overflowY: "auto",
      fontSize: "14px",
    }}
  >
    <h3 style={{ marginBottom: "10px", color: "#007bba" }}>
      Location Insights
    </h3>

    {prediction && (
      <div
        style={{
          fontSize: "24px",
          fontWeight: "bold",
          color: "#007bba",
          marginBottom: "12px",
        }}
      >
        Predicted Trips per Dock: {Math.round(prediction).toLocaleString()}
      </div>
    )}

    <div><b>Transit Stops (275m):</b> {featureResults.count_transit_stop_275m}</div>
    <div><b>Amenities (275m):</b> {featureResults.count_amenities_275m}</div>
    <div><b>Jobs (275m):</b> {featureResults.jobs_count_within_275m}</div>

    <div style={{ marginTop: "8px" }}>
      <b>Nearest Station:</b>{" "}
      {Math.round(featureResults.nearest_bikeshare_station_m)} m
    </div>

    <div>
      <b>Population Density:</b>{" "}
      {featureResults.population_density.toFixed(4)}
    </div>

    {topSummary.length > 0 && (
      <div style={{ marginTop: "16px" }}>
        <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
          Why this score?
        </h4>

        {topSummary.slice(0, 5).map((row) => (
          <div
            key={row.feature}
            style={{
              marginBottom: "8px",
              padding: "8px",
              borderRadius: "8px",
              background: "#f4f7f9",
              borderLeft: `4px solid ${
                row.shap_value >= 0 ? "#007bba" : "#999"
              }`,
            }}
          >
            <div><b>{row.feature}</b></div>

            <div>
              {row.shap_value >= 0 ? "Increases" : "Decreases"} prediction
            </div>

            <div style={{ fontSize: "12px", color: "#555" }}>
              percentile: {Math.round(row.percentile_rank)} ·{" "}
              {row.relative_to_median}
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
)}  

      <MapContainer
        center={AUSTIN_CENTER}
        zoom={10}
        style={{ height: "100%", width: "100%" }}
      >
        {/* Carto Positron basemap */}
        <TileLayer
          attribution='&copy; OpenStreetMap &copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
        />

        {/* outside mask */}
        {countyHoles.length > 0 && (
          <Polygon
            positions={[OUTER_RING, ...countyHoles]}
            pathOptions={{
              stroke: false,
              fillColor: "#f2f2f2",
              fillOpacity: 0.65,
              fillRule: "evenodd",
              interactive: false,
            }}
          />
        )}

        {/* Travis County border */}
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

        {/* click marker */}
        <ClickPoint
          selectedPoint={selectedPoint}
          setSelectedPoint={setSelectedPoint}
        />
      </MapContainer>
    </div>
  );
}
