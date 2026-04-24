import "leaflet/dist/leaflet.css";
import L from "leaflet";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Polygon,
  CircleMarker,
  Popup,
  useMap,
  useMapEvents,
} from "react-leaflet";
import { useEffect, useMemo, useRef, useState } from "react";
import StartScreen from "./components/StartScreen";
import HintsButton from "./components/HintsButton";
import ContextLayers from "./components/ContextLayers";
import StationResults, { ScoringOverlay } from "./components/StationResults";

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
  if (!geojson?.features?.length) return [];

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
  if (scale === 1) return rings;

  return rings.map((ring) =>
    ring.map(([lat, lng]) => [
      center.lat + (lat - center.lat) * scale,
      center.lng + (lng - center.lng) * scale,
    ])
  );
}

function isPointInRing(point, ring) {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];

    const intersects =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;

    if (intersects) inside = !inside;
  }

  return inside;
}

function isPointInPolygonCoords(point, polygonCoords) {
  if (!polygonCoords?.length) return false;

  if (!isPointInRing(point, polygonCoords[0])) return false;

  for (let i = 1; i < polygonCoords.length; i++) {
    if (isPointInRing(point, polygonCoords[i])) return false;
  }

  return true;
}

function geometryContainsPoint(geometry, point) {
  if (!geometry) return false;

  if (geometry.type === "Polygon") {
    return isPointInPolygonCoords(point, geometry.coordinates);
  }

  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.some((polygonCoords) =>
      isPointInPolygonCoords(point, polygonCoords)
    );
  }

  return false;
}

function geoJsonContainsPoint(geojson, point) {
  if (!geojson) return false;

  if (geojson.type === "FeatureCollection") {
    return geojson.features?.some((feature) =>
      geometryContainsPoint(feature.geometry, point)
    );
  }

  if (geojson.type === "Feature") {
    return geometryContainsPoint(geojson.geometry, point);
  }

  return geometryContainsPoint(geojson, point);
}

// -----------------------------
// Map helpers
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

function MapInteractionWatcher({ enabled, shouldIgnoreInteraction, onInteract }) {
  function handleMapInteraction() {
    if (!enabled) return;
    if (shouldIgnoreInteraction?.()) return;

    onInteract();
  }

  useMapEvents({
    dragstart: handleMapInteraction,
    zoomstart: handleMapInteraction,
  });

  return null;
}

function PanSelectedPointIntoView({
  selectedPoint,
  enabled,
  verticalPosition = 0.06,
  zoomBoost = 1,
  maxZoom = 15,
  onAutoPanStart,
  onAutoPanEnd,
}) {
  const map = useMap();

  const onAutoPanStartRef = useRef(onAutoPanStart);
  const onAutoPanEndRef = useRef(onAutoPanEnd);

  useEffect(() => {
    onAutoPanStartRef.current = onAutoPanStart;
    onAutoPanEndRef.current = onAutoPanEnd;
  }, [onAutoPanStart, onAutoPanEnd]);

  useEffect(() => {
    if (!enabled || !selectedPoint) return;

    const timeout = window.setTimeout(() => {
      onAutoPanStartRef.current?.();

      const currentZoom = map.getZoom();
      const targetZoom = Math.min(currentZoom + zoomBoost, maxZoom);
      const size = map.getSize();

      const selectedProjectedPoint = map.project(
        [selectedPoint.lat, selectedPoint.lon],
        targetZoom
      );

      const mapCenterPixel = L.point(size.x / 2, size.y / 2);
      const desiredPointPixel = L.point(size.x / 2, size.y * verticalPosition);

      const newCenterProjectedPoint = selectedProjectedPoint
        .add(mapCenterPixel)
        .subtract(desiredPointPixel);

      const newCenterLatLng = map.unproject(newCenterProjectedPoint, targetZoom);

      let hasEnded = false;

      function endAutoPan() {
        if (hasEnded) return;
        hasEnded = true;
        onAutoPanEndRef.current?.();
      }

      map.once("moveend", endAutoPan);

      map.flyTo(newCenterLatLng, targetZoom, {
        animate: true,
        duration: 0.9,
        easeLinearity: 0.25,
      });

      window.setTimeout(endAutoPan, 1400);
    }, 240);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [enabled, selectedPoint, verticalPosition, zoomBoost, maxZoom, map]);

  return null;
}

// -----------------------------
// Floating controls
// -----------------------------
function ScoreButton({
  selectedPoint,
  isLoading,
  hasScoredStation,
  onScore,
}) {
  if (!selectedPoint || hasScoredStation) return null;

  return (
    <button
      className={`score-location-button ${isLoading ? "is-loading" : ""}`}
      onClick={onScore}
      disabled={isLoading}
    >
      {isLoading ? "Scoring your station..." : "Score My Station"}
    </button>
  );
}

function SelectedPointCard({ selectedPoint, hasScoredStation }) {
  if (!selectedPoint || hasScoredStation) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: "20px",
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
  const [hasScoredStation, setHasScoredStation] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);
  const [shouldAutoPanResults, setShouldAutoPanResults] = useState(false);

  const [showBoundaryWarning, setShowBoundaryWarning] = useState(false);
  const [boundaryWarningKey, setBoundaryWarningKey] = useState(0);

  const boundaryWarningTimeoutRef = useRef(null);
  const ignoreMapInteractionRef = useRef(false);

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

  useEffect(() => {
    return () => {
      if (boundaryWarningTimeoutRef.current) {
        window.clearTimeout(boundaryWarningTimeoutRef.current);
      }
    };
  }, []);

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
    setHasScoredStation(false);
    setResultsCollapsed(false);
    setShouldAutoPanResults(false);
    setShowBoundaryWarning(false);
    ignoreMapInteractionRef.current = false;
  }

  function triggerBoundaryWarning() {
    if (boundaryWarningTimeoutRef.current) {
      window.clearTimeout(boundaryWarningTimeoutRef.current);
    }

    setBoundaryWarningKey((prev) => prev + 1);
    setShowBoundaryWarning(true);

    boundaryWarningTimeoutRef.current = window.setTimeout(() => {
      setShowBoundaryWarning(false);
    }, 2550);
  }

  function handlePointSelect(point) {
    const isInsideCounty = geoJsonContainsPoint(countyGeoJson, [
      point.lon,
      point.lat,
    ]);

    if (!isInsideCounty) {
      if (!selectedPoint) {
        triggerBoundaryWarning();
      }

      return;
    }

    setShowBoundaryWarning(false);

    setSelectedPoint(point);
    setHasScoredStation(false);
    setResultsCollapsed(false);
    setShouldAutoPanResults(false);
    ignoreMapInteractionRef.current = false;

    setFeatureResults(null);
    setPrediction(null);
    setTopSummary([]);
    setStationComparison(null);
  }

  function handleTryAnother() {
    setSelectedPoint(null);
    setFeatureResults(null);
    setPrediction(null);
    setTopSummary([]);
    setStationComparison(null);
    setHasScoredStation(false);
    setResultsCollapsed(false);
    setIsLoading(false);
    setShouldAutoPanResults(false);
    setShowBoundaryWarning(false);
    ignoreMapInteractionRef.current = false;
  }

  async function getFeatures() {
    if (!selectedPoint) {
      alert("Click the map first!");
      return;
    }

    setHasScoredStation(true);
    setResultsCollapsed(false);
    setShouldAutoPanResults(false);
    setIsLoading(true);
    ignoreMapInteractionRef.current = false;

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
      setResultsCollapsed(false);
      setShouldAutoPanResults(true);
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
              <SelectedPointCard
                selectedPoint={selectedPoint}
                hasScoredStation={hasScoredStation}
              />

              {isLoading && <ScoringOverlay />}

              <StationResults
                featureResults={featureResults}
                prediction={prediction}
                stationComparison={stationComparison}
                topSummary={topSummary}
                isCollapsed={resultsCollapsed}
                onExpand={() => {
                  ignoreMapInteractionRef.current = true;
                  setResultsCollapsed(false);

                  window.setTimeout(() => {
                    ignoreMapInteractionRef.current = false;
                  }, 400);
                }}
                onTryAnother={handleTryAnother}
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

                <ContextLayers
                  selectedPoint={selectedPoint}
                  showLayers={Boolean(featureResults) && !isLoading}
                  radiusMeters={200}
                />

                <MapInteractionWatcher
                  enabled={Boolean(featureResults) && !resultsCollapsed}
                  shouldIgnoreInteraction={() => ignoreMapInteractionRef.current}
                  onInteract={() => setResultsCollapsed(true)}
                />

                <PanSelectedPointIntoView
                  selectedPoint={selectedPoint}
                  enabled={
                    shouldAutoPanResults &&
                    Boolean(featureResults) &&
                    !isLoading &&
                    !resultsCollapsed
                  }
                  verticalPosition={0.25}
                  zoomBoost={1}
                  maxZoom={15}
                  onAutoPanStart={() => {
                    ignoreMapInteractionRef.current = true;
                  }}
                  onAutoPanEnd={() => {
                    setShouldAutoPanResults(false);

                    window.setTimeout(() => {
                      ignoreMapInteractionRef.current = false;
                    }, 150);
                  }}
                />

                <ClickPoint
                  selectedPoint={selectedPoint}
                  onPointSelect={handlePointSelect}
                />
              </MapContainer>
            </div>
          </div>

          <HintsButton className="map-screen-hints" />

          <ScoreButton
            selectedPoint={selectedPoint}
            isLoading={isLoading}
            hasScoredStation={hasScoredStation}
            onScore={getFeatures}
          />
        </div>
      )}

      {showStartScreen && (
        <StartScreen
          onPlayStart={handlePlayStart}
          onTransitionComplete={() => setShowStartScreen(false)}
        />
      )}

      {showBoundaryWarning && !selectedPoint && (
        <div key={boundaryWarningKey} className="boundary-warning-toast">
          Please select a point within Travis County
        </div>
      )}
    </div>
  );
}