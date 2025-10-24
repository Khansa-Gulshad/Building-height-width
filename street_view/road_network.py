# Libraries for working with maps and geospatial data
import geopandas as gpd
import osmnx as ox
import numpy as np

def get_road_network(city):
    cf = '["highway"~"primary|secondary|tertiary|residential|primary_link|secondary_link|tertiary_link|living_street|service|unclassified"]'
    G = ox.graph_from_place(city, simplify=True, custom_filter=cf)
    # (duplicate-edge removal optional)
    G_proj = ox.project_graph(G)
    _, edges = ox.graph_to_gdfs(G_proj)
    return edges


# Get a list of points over the road map with a N distance between them
def select_points_on_road_network_projected(roads, N=50):
    """Sample points every N meters in the SAME (projected) CRS as `roads`."""
    points = []
    for row in roads.itertuples(index=True, name='Road'):
        geom = row.geometry
        if geom is None or not hasattr(geom, "length"):
            continue
        L = float(geom.length)
        for d in range(0, int(L) + 1, N):
            points.append([geom.interpolate(d), row.Index])
    gdf = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry", crs=roads.crs)
    gdf = gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    gdf["id"] = np.arange(len(gdf), dtype=int)
    return gdf  # STILL projected
