# Libraries for working with maps and geospatial data
import geopandas as gpd
import osmnx as ox
import numpy as np
import math  # (not strictly needed now, but fine to keep)

# Generates a road network from either a placename or bounding box using OpenStreetMap data
# Same as your version, but NO road_angle computed or attached.
def get_road_network(city, bbox=None):
    print(f"Fetching road network for city: {city}")
    cf = '["highway"~"primary|secondary|tertiary|residential|primary_link|secondary_link|tertiary_link|living_street|service|unclassified"]'

    try:
        if bbox:
            G = ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], simplify=True, custom_filter=cf)
        else:
            G = ox.graph_from_place(city, simplify=True, custom_filter=cf)
    except Exception:
        return gpd.GeoDataFrame()

    # collapse reverse duplicates: keep one direction per physical segment
    unique_roads = set()
    G_simplified = G.copy()
    for u, v, key in list(G.edges(keys=True)):
        if (v, u) in unique_roads:
            G_simplified.remove_edge(u, v, key)
        else:
            unique_roads.add((u, v))
    G = G_simplified

    # project to a metric CRS (meters) and return edges GeoDataFrame
    G_proj = ox.project_graph(G)
    _, edges = ox.graph_to_gdfs(G_proj)
    return edges


# Get a gdf of points over a road network with a N distance between them (in projected CRS)
def select_points_on_road_network(roads, N=50):
    N = max(1, int(N))
    points = []

    for row in roads.itertuples(index=True, name='Road'):
        linestring = row.geometry
        if linestring is None:
            continue
        length = float(linestring.length)
        # 0, N, 2N, ..., < length  (match your original behavior: end not guaranteed)
        for distance in range(0, int(length), N):
            point = linestring.interpolate(distance)
            points.append([point, row.Index])

    gdf_points = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry")
    gdf_points.set_crs(roads.crs, inplace=True)
    gdf_points = gdf_points.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    return gdf_points
