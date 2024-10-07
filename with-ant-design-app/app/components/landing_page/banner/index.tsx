import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Divider } from "antd";
import styles from "./Banner.module.css"; // Ensure you create a CSS module for styling

import { Parallax } from "rc-scroll-anim";

const Banner = () => {
  const numberOfObjects = 60; // Define the number of cubes

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

    const rect = bannerRef.current.getBoundingClientRect();
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const cubesSettings = useMemo(() => {
    const generateBellCurvePosition = (size: number) => {
      const position = Math.random();
      const distanceFromSide = (Math.exp(-5 * position) * size) / 2;

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
      const layer = Math.floor(Math.random() * 5);
      const speed = (layer * (layer / 2)) * 0.005;
      const initialX = generateBellCurvePosition(bannerSize.width);
      const initialY = generateUniformPosition(bannerSize.height);

      //   console.log(bannerSize);
      //   console.log({ layer, speed, initialX, initialY });

      return { layer, speed, initialX, initialY };
    });
  }, [bannerSize.width, bannerSize.height]);

  const cubes = cubesSettings.map((cube, index) => {
    const { layer, speed, initialX, initialY } = cube;
    const x = (mousePosition.x - bannerSize.width / 2) * speed + initialX;
    const y = (mousePosition.y - bannerSize.height / 2) * speed + initialY;

    return (
      <div
        key={index}
        style={{
          position: "absolute",
          width: "3px",
          height: "3px",
          backgroundColor: "rgba(255,255,255,0.8)",
          left: `calc(50% + ${x}px)`,
          top: `calc(50% + ${y}px)`,
          transform: `translate(-50%, -50%) scale(${layer})`,
        }}
      />
    );
  });

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
          height: "80vh",
          background: "linear-gradient(to right, #1a1a1a, #3e236e)", // Single color gradient
        }}
      >
        {cubes}
        <div
          className={styles.content}
        >
          <div className={styles.title}>Inovācija cenu noteikšanā</div>
          <Divider 
            style={{
              color: "white",
              borderColor: "white",
              // width: "50%",
              margin: "10px auto",
            }}
          />
          <div className={styles.subtitle}>Nosaki īpašuma cenu ātri un vienkārši</div>
          {/* <div>
            <Button type="primary" size="large">
              Learn More
            </Button>
            <Button size="large" style={{ marginLeft: 20 }}>
              Contact Us
            </Button>
          </div> */}
        </div>
      </div>
    // </Parallax>
  );
};

export default Banner;
