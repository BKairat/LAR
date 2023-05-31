"""
The main Python file which used for detecting the objects of certain colors ([68, 59, 102] RBG format),
and asle capable to predict at what angle the robot took the picture.
"""
import cv2
from image_analysis import analysisRGB, getAngleToColor, isEntry
from map import Map


if __name__ == "__main__":
        img_path = "map_images/sc"+"3"+".png"
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mapa = Map()
        mapa.mapClear()
        garage, obstacles = analysisRGB(img)
        # mapa.mapClear()

        for i in obstacles:
            for j in i:
                mapa.drawObstacle(j.dist, j.angleMap, j.color)
                cv2.drawContours(img, [j.contour], 0, (0, 255, 0), 1)
        if garage.angleP:
            mapa.drawGarage(garage.dist[0], garage.angleMap[0], garage.angleP, garage.entry)
            for i in garage.contour:
                cv2.drawContours(img, [i], -1, (0, 255, 0), 1)
        elif garage.entry:
            print(garage.dist, garage.angleMap, garage.angleP, garage.entry)
            mapa.drawGarage(garage.dist, garage.angleMap, garage.angleP, garage.entry)
        else:
            mapa.drawGarage(garage.dist, garage.angleMap, garage.angleP, garage.entry)

        dist, rot = mapa.bfs()
        print(dist, "\n", rot)
        mapa.drawRobot()


        while True:
            cv2.imshow("file", img)  # shows image
            if cv2.waitKey(0) & 0xFF == ord('q'):
                # mapa.drawCircle(200, 200, 210, "WHITE")
                # del mapa
                # mapa = Map()
                break

            if cv2.waitKey(0) & 0xFF == ord('m'):
                mapa.showMap()
                # mapa.drawCircle(200, 200, 210, "WHITE")
                # del mapa
                # mapa = Map()
                break
        # mapa.mapClear()
