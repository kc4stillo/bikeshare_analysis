import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { useEffect, useState } from "react";

const AUSTIN_CENTER = [30.2672, -97.7431];

export default function App() {
  const [countyGeoJson, setCountyGeoJson] = useState(null);

  useEffect(() => {
    fetch("/travis_county.geojson")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log("Loaded GeoJSON:", data);
        setCountyGeoJson(data);
      })
      .catch((err) => {
        console.error("Error loading GeoJSON:", err);
      });
  }, []);

  return (
    <div style={{ height: "100vh", width: "100%" }}>
      <MapContainer
        center={AUSTIN_CENTER}
        zoom={10}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {countyGeoJson && (
          <GeoJSON
            data={countyGeoJson}
            style={() => ({
              color: "red",
              weight: 5,
              fillColor: "yellow",
              fillOpacity: 0.15,
            })}
          />
        )}
      </MapContainer>
    </div>
  );
}