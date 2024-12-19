"use client";

import React, { useEffect, useRef, useState } from "react";
import { Button, Row, Col } from "antd";


import { useRouter } from "next/navigation";

import { Carousel, Radio } from "antd";

import TweenOne from "rc-tween-one";
import Children from "rc-tween-one/lib/plugin/ChildrenPlugin";

import { OverPack, Parallax } from "rc-scroll-anim";

import styles from "./Statistics.module.css";
import QueueAnim from "rc-queue-anim";
import { DotPosition } from "antd/es/carousel";
import Link from "next/link";
import { AreaChartOutlined, BookOutlined, DatabaseOutlined, ExportOutlined } from "@ant-design/icons";


import { tsParticles } from "@tsparticles/engine";
import { loadFull } from "tsparticles";
import Particles from "@tsparticles/react";


TweenOne.plugins.push(Children);

const buttonContents = [
  { label: "Uzzini vairāk par platformā izmantotajiem ", value: 15000, labelLink: "/data", labelLinkText: " datu ierakstiem", labelLinkExternal: true,
    icon: <DatabaseOutlined /> },
  { label: "Izpēti ", value: 189, labelLink: "/groups", labelLinkText: " publiski pieejamos sludinājumus",
    icon: <BookOutlined />
  },
  { label: "Apskati ", value: 10, labelLink: "/groups", labelLinkText: " izmantotos sludinājumu veidus",
    icon: <AreaChartOutlined />
  },
];



async function loadParticles(options: any) {
  await loadFull(tsParticles);
  console.log("tsParticles loaded");
  return await tsParticles.load({ options });
}

const configs = {
  particles: {
    move: {
      angle: {
        value: 15
      },
      enable: true,
      direction: "top",
      outModes: "destroy"
    },
    shape: {
      type: "image",
      options: {
        image: {
          src:
            "https://icons.iconarchive.com/icons/designbolts/cute-social-2014/256/Google-Plus-One-icon.png",
          width: 256,
          height: 256
        }
      }
    },
    size: {
      value: 32
    }
  }
};

const Statistics: React.FC = () => {
  const [value, setValue] = useState(buttonContents[0].value);

  const [values, setValues] = useState([0, 0, 0]);

  const router = useRouter();

  const likeBtnRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    async function loadParticlesConfig() {
      const container = await loadParticles(configs);
      console.log("Particles loaded", container);
  
      if (likeBtnRef.current && container) {
        likeBtnRef.current.addEventListener("mouseover", () => {
          const rect = likeBtnRef.current!.getBoundingClientRect();
  
          if (container && container.particles) {
            container.particles.addParticle({
              x: (rect.left + Math.random() * rect.width) * container.retina.pixelRatio,
              y: (rect.top + Math.random() * rect.height) * container.retina.pixelRatio,
            });
          }
        });
      }
    }
  
    loadParticlesConfig();
  }, []);
  
  const [dotPosition, setDotPosition] = useState<DotPosition>("top");

  const handleClick = (link: string) => {
    // setValue((prevValue) => prevValue + 1000);
    // setValues((prevValues) => prevValues.map((value) => value + 1000));

    router.push(link); // Redirect after animationme
  };

  return (
    <div className={styles["statistics"]}>
        <OverPack
          style={{
            overflow: "hidden",
            height: "100%",
            // width: "70%",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <QueueAnim
            key="queue"
            leaveReverse
            className={styles["statistic-container"]}
            
            onEnd={(e) => {
              // setValue((prev) => prev + 1000);
              // setValues(buttonContents.map((item) => item.value));
              if (e.type == "enter" && e.key == 0) {
                setValue(buttonContents[0].value);
                setValues(buttonContents.map((item) => item.value));
              }
              if (e.type == "leave") {
                setValue(0);
                setValues([0, 0, 0]);
              }
            }}
            >
            {buttonContents.map((buttonContent, index) => ( 
              <div id="like" key={index} className={styles["statistic"]} ref={likeBtnRef}>
                <div className={`${styles["statistic-button"]} ${styles[`statistic-button-type-${index + 1}`]}`} onClick={()=>handleClick(buttonContent.labelLink)}>
                  {/* {buttonContent.icon} */}
                  <TweenOne
                    animation={{
                      Children: {
                        value: values[index],
                        // value: value,
                        floatLength: 0,
                        formatMoney: true,
                      },
                      delay: 0,
                      duration: 1500,
                    }}
                    className={styles.numberdisplay}
                    style={{
                      width: "fit-content",
                    }}
                  />
                </div>
                <QueueAnim
                  key="queue1"
                  // type="bottom"
                  className={styles["statistic-label"]}>
                
                <div className={styles["statistic-label-text"]}>
                  
                  {buttonContent.label}
                  
                  {buttonContent.labelLinkText}

                  <div className={styles["statistic-label-link"]}>
                    
                  {buttonContent.labelLinkExternal && <ExportOutlined />}
                    
                  {/* {buttonContent.labelLink && (
                    <Link href={buttonContent.labelLink} className={styles["statistic-label-link-group"]}>
                      {buttonContent.labelLinkText}
                    </Link>  
                  )} */}
                  </div>

                  </div>
                  
                </QueueAnim>
              </div>
            ))}
          </QueueAnim>
        </OverPack>
    </div>
  );
};

export default Statistics;
