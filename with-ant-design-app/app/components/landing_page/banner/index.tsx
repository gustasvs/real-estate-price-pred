"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Divider } from "antd";
import styles from "./Banner.module.css"; // Ensure you create a CSS module for styling

import { OverPack, Parallax } from "rc-scroll-anim";
import { useRouter, useSearchParams } from "next/navigation";
import { FaArrowRight, FaHouseChimney, FaHouseChimneyUser, FaHouseLaptop } from "react-icons/fa6";
import { BiArrowFromLeft, BiArrowToLeft, BiBuildingHouse, BiSolidRightArrow } from "react-icons/bi";
import QueueAnim from "rc-queue-anim";
import { useSession } from "next-auth/react";
import { useThemeContext } from "../../../context/ThemeContext";

const Banner = () => {

  const router = useRouter();

  const { theme } = useThemeContext();

  const { data: session, status, update } = useSession();

  const searchParams = useSearchParams();

  const numberOfObjects = 90;

  const bannerRef = useRef<HTMLDivElement>(null);
  const [bannerSize, setBannerSize] = useState({ width: 0, height: 0 });

  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const updateSize = () => {
      if (!!bannerRef.current) {
        setBannerSize({
          width: bannerRef.current.offsetWidth,
          height: bannerRef.current.offsetHeight,
        });
      }
    };
    window.addEventListener("resize", updateSize);
    updateSize(); // Initial call
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const handleMouseMove = (e: { clientX: number; clientY: number }) => {
    if (!bannerRef.current) return;

    // const rect = bannerRef.current.getBoundingClientRect();
    // rect is all window
    const rect = { left: 0, top: 0 };
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const cubesSettings = useMemo(() => {
    const generateBellCurvePosition = (size: number) => {
      const position = Math.random();
      const distanceFromSide = (Math.exp(-6 * position) * size) / 2;

      const absoluteDistance =
        Math.random() > 0.5
          ? size / 2 - distanceFromSide
          : (size / 2) * -1 + distanceFromSide;

      // console.log({ position, distanceFromSide, absoluteDistance });
      if (absoluteDistance < 250 && absoluteDistance > -250) {
        return absoluteDistance * 1.7;
      }
      return absoluteDistance;
    };

    const generateUniformPosition = (size: number) => {
      return Math.random() * size - size / 2;
    };

    return Array.from({ length: numberOfObjects }, () => {
      const layer = Math.floor(Math.random() * 4);
      const speed = (layer * (layer / 2)) * 0.005;
      const initialX = generateBellCurvePosition(bannerSize.width);
      const initialY = generateUniformPosition(bannerSize.height);

      const randomClass = Math.floor(Math.random() * 8);
      const cubeClass = `cube-${randomClass}`;

      //   console.log(bannerSize);
      //   console.log({ layer, speed, initialX, initialY });

      return { layer, speed, initialX, initialY, cubeClass };
    });
  }, [bannerSize.width, bannerSize.height]);

  const cubes = cubesSettings.map((cube, index) => {
    const { layer, speed, initialX, initialY, cubeClass } = cube;
    const x = (mousePosition.x - bannerSize.width / 2) * speed + initialX;
    const y = (mousePosition.y - bannerSize.height / 2) * speed + initialY;

    // const currentX = initialX + (x - initialX) * 0.1;
    // const currentY = initialY + (y - initialY) * 0.1;

    return (
      <div
        key={index}
        style={{
          position: "absolute",
          width: "0.5rem",
          height: "0.5rem",
          left: `calc(50% + ${x}px)`,
          top: `calc(50% + ${y}px)`,
          opacity: layer / 5,
          transform: `translate(-50%, -50%)`,
          animationDelay: `${index * 0.5}s`, // Delay based on index, modify as needed
          filter: theme === "dark" ? "invert(1)" : "invert(0)",
          // transition: "left 0.1s linear, top 0.1s linear",
          ...(layer && { '--scale': (3 + (layer / 2)).toString() } as React.CSSProperties)
        }}
        className={`${styles[cubeClass]} ${styles.wander} ${styles.cube}`}
      />
    );
  });

  const getStartedText = session?.user?.email ? "Sāc darbu pievienojot savu pirmo objektu!" : "Sāc darbu izveidojot savu kontu!";
// 
  return (
    // <Parallax
    //   // animation={[{ y: 100, opacity: 0, playScale: [0.9, 1] }]}
    //   style={{ transform: "translateY(0px)", opacity: 1 }}
    // >
    <div
      ref={bannerRef}
      className={styles.banner}
      onMouseMove={handleMouseMove}
      style={{
        // background: "linear-gradient(to right, #1a1a1a, #3e236e)", // Single color gradient
      }}
    >
      {cubes}
      <div
        className={styles.content}
      >
        <div className={styles["left-container"]}>
          <div className={styles.title}>SmartEstate</div>
          <div style={{
            display: "flex",
            width: "80%",
            marginLeft: "20%",
            paddingTop: "10px",
            paddingBottom: "10px",
          }}>
            <Divider
              style={{
                color: "white",
                borderColor: "white",
                margin: "10px auto",
              }}
            />
          </div>
          <div className={styles.subtitle}>
            Saglabā savus iecienītākos nekustamos īpašumus <br></br>
            un iegūsti to īsto cenu izmantojot modernus <br></br>
            mašīnmācīšanās algoritmus
          </div>
        </div>
        <div
          style={{
            display: "flex",
            height: "70vh",
            marginLeft: "2em",
          }}
        >
          <Divider
            type="vertical"
            style={{
              display: "flex",
              color: "white",
              borderColor: "white",
              height: "100%",
            }}
          />
        </div>
        <div className={styles["right-container"]}>
          <div className={styles["right-container-items"]}>
            <OverPack>
              
              <BiBuildingHouse className={styles["get-started-icon"]} />
            </OverPack>
            <div
              className={styles["get-started-button"]}
              onClick={() => {
                if (session?.user?.email) {
                  router.push("/groups");
                } else {
                  const params = new URLSearchParams(searchParams);
                  params.set("modal", "sign-up");

                  router.replace(`?${params.toString()}`, { scroll: false });
                }
              }}
            >
              <span className={styles["get-started-text"]}>
                {getStartedText}
              </span>
              <FaArrowRight className={styles["get-started-next-arrow"]} />
              <FaArrowRight className={styles["get-started-prev-arrow"]} />
            </div>

          </div>
        </div>
      </div>
    </div>
    // </Parallax>
  );
};

export default Banner;
