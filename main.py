import lavavu

from spherical.spherical import st_gen

def main():
    # app_dir = os.path.join(os.getcwd(), "TEMPORAL")
    # if not os.path.exists(app_dir):
    #     os.makedirs(app_dir)
    # print(app_dir)

    dataDir = "C:/Cartography/DIP/Project/Data/"
    borderShp = 'C:\Cartography\DIP\Project\Data\shpEast\Border.shp'
    proj_crs_str = '+proj=laea +lon_0=181.5820313 +lat_0=90 +datum=WGS84 +units=m +no_defs'

    st = st_gen(data_src=dataDir + 'вост-арктика_в атлас.dat', border_src=borderShp, lon_type='360')

    lv = lavavu.Viewer(border=False, resolution=[400,400], background="#FFFFFF")
    lv["axis"]=True
    lv['specular'] = 0.7

    tris = lv.triangles("coarse",  wireframe=False, colour="#4444FF", opacity=0.8)
    tris.vertices(st.points)
    tris.indices(st.simplices)

    trisw = lv.triangles("coarsew",  wireframe=True, colour="#000000", opacity=0.8, linewidth=3.0)
    trisw.vertices(st.points)
    trisw.indices(st.simplices)

    lv.display()
    lv.image(r"C:\Cartography\DIP\Project\Code\test4.png", resolution=(5000,5000))

if __name__ == "__main__":
    main()
