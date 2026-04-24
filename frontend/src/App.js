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

const OUTER_RING = [
  [90, -180],
  [90, 180],
  [-90, 180],
  [-90, -180],
];

function getCountyOuterRings(geojson) {
  if (!geojson || !geojson.features || geojson.features.length === 0) {
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
function StationComparisonChart({ stationComparison }) {
  if (!stationComparison || !stationComparison.all_station_rankings) return null;

  const rows = stationComparison.all_station_rankings;
  const maxTrips = Math.max(...rows.map((row) => row.trips_per_dock));

  return (
    <div style={{ marginTop: "16px" }}>
      <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
        Station Ranking
      </h4>

      <div
        style={{
          marginBottom: "10px",
          padding: "10px",
          background: "#f4f7f9",
          borderRadius: "8px",
          borderLeft: "4px solid #007bba",
        }}
      >
        <div>
          <b>Rank:</b> {stationComparison.rank_position} of{" "}
          {stationComparison.total_stations_plus_candidate}
        </div>
        <div>
          <b>Percentile:</b>{" "}
          {stationComparison.rank_percentile.toFixed(1)}%
        </div>
      </div>

      <div
        style={{
          maxHeight: "260px",
          overflowY: "auto",
          paddingRight: "4px",
        }}
      >
        {rows.map((row) => {
          const widthPercent = (row.trips_per_dock / maxTrips) * 100;

          return (
            <div
              key={`${row.rank}-${row.name}`}
              style={{
                marginBottom: "7px",
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
                  fontSize: "12px",
                  marginBottom: "2px",
                  fontWeight: row.is_candidate ? "bold" : "normal",
                }}
              >
                <span>
                  #{row.rank} {row.is_candidate ? "Your location" : row.name}
                </span>
                <span>{Math.round(row.trips_per_dock).toLocaleString()}</span>
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

export default function App() {
  const [countyGeoJson, setCountyGeoJson] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [featureResults, setFeatureResults] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [topSummary, setTopSummary] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [stationComparison, setStationComparison] = useState(null);

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

  const countyHoles = useMemo(() => {
    return getCountyOuterRings(countyGeoJson);
  }, [countyGeoJson]);

  async function getFeatures() {
    console.log("Get Features button clicked");

    if (!selectedPoint) {
      alert("Click the map first!");
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(
        "https://bikeshare-analysis.onrender.com/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            lat: selectedPoint.lat,
            lon: selectedPoint.lon,
          }),
        }
      );

      console.log("Response status:", response.status);

      const data = await response.json();
      console.log("Prediction response:", data);

      if (data.success) {
        setFeatureResults(data.features);
        setPrediction(data.predicted_trips_per_dock);
        setTopSummary(data.top_feature_summary || []);
        setStationComparison(data.station_comparison);
      } else {
        console.error("Backend returned error:", data.error);
        alert(`Backend error: ${data.error}`);
      }
    } catch (error) {
      console.error("Fetch failed:", error);
      alert("Could not reach backend. Check the browser console.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div style={{ height: "100vh", width: "100%" }}>
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
          borderRadius: "8px",
          cursor: "pointer",
          fontWeight: "bold",
          boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
        }}
      >
        {isLoading ? "Scoring..." : "Get Features"}
      </button>

      {selectedPoint && (
        <div
          style={{
            position: "absolute",
            top: "70px",
            right: "20px",
            zIndex: 1000,
            background: "white",
            padding: "10px 12px",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            fontSize: "13px",
          }}
        >
          <div>
            <b>Lat:</b> {selectedPoint.lat}
          </div>
          <div>
            <b>Lon:</b> {selectedPoint.lon}
          </div>
        </div>
      )}

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
      width: "370px",
      maxHeight: "78vh",
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
          lineHeight: "1.15",
        }}
      >
        Predicted Trips per Dock:{" "}
        {Math.round(prediction).toLocaleString()}
      </div>
    )}

    {stationComparison && (
      <div
        style={{
          marginBottom: "14px",
          padding: "10px",
          background: "#f4f7f9",
          borderRadius: "8px",
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
          Percentile: {stationComparison.rank_percentile.toFixed(1)}%
        </div>
      </div>
    )}

    {stationComparison?.all_station_rankings && (
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
          {stationComparison.all_station_rankings.map((row) => {
            const maxTrips = Math.max(
              ...stationComparison.all_station_rankings.map(
                (station) => station.trips_per_dock
              )
            );

            const widthPercent = Math.max(
              3,
              (row.trips_per_dock / maxTrips) * 100
            );

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
                    #{row.rank}{" "}
                    {row.is_candidate ? "Your location" : row.name}
                  </span>

                  <span>
                    {Math.round(row.trips_per_dock).toLocaleString()}
                  </span>
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
    )}

    <div
      style={{
        marginBottom: "14px",
        padding: "10px",
        background: "#fafafa",
        borderRadius: "8px",
      }}
    >
      <h4 style={{ marginBottom: "8px", color: "#007bba" }}>
        Nearby Attributes
      </h4>

      <div>
        <b>Transit Stops (275m):</b>{" "}
        {featureResults.count_transit_stop_275m}
      </div>

      <div>
        <b>Amenities (275m):</b>{" "}
        {featureResults.count_amenities_275m}
      </div>

      <div>
        <b>Jobs (275m):</b>{" "}
        {featureResults.jobs_count_within_275m}
      </div>

      <div style={{ marginTop: "8px" }}>
        <b>Nearest Station:</b>{" "}
        {Math.round(featureResults.nearest_bikeshare_station_m)} m
      </div>

      <div>
        <b>Population Density:</b>{" "}
        {featureResults.population_density?.toFixed(4)}
      </div>
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
            <div>
              <b>{row.feature}</b>
            </div>

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
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
          maxZoom={20}
        />

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
          setSelectedPoint={setSelectedPoint}
        />
      </MapContainer>
    </div>
  );
}