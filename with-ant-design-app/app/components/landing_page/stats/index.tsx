import React, { useEffect, useState } from "react";
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

TweenOne.plugins.push(Children);

const buttonContents = [
  { label: "Algoritmam", value: 15000, labelLink: "/data", labelLinkText: "izmantotie dati" },
  { label: "Platformā pieejamie", value: 200, labelLink: "/data", labelLinkText: "sludinājumi" },
  { label: "Sludinājumu", value: 10, labelLink: "/groups", labelLinkText: "veidi" },
];

const Statistics: React.FC = () => {
  const [value, setValue] = useState(buttonContents[0].value);

  const [values, setValues] = useState([0, 0, 0]);

  useEffect(() => {
    console.log("values", values);  
  }, [values]);

  const [dotPosition, setDotPosition] = useState<DotPosition>("top");

  const handleClick = () => {
    setValue((prevValue) => prevValue + 1000);
    setValues((prevValues) => prevValues.map((value) => value + 1000));
  };

  return (
    <div className={styles["statistics"]}>
      <Carousel dotPosition={dotPosition} style={{}}>
        <OverPack
          style={{
            overflow: "hidden",
            height: "100vh",
            width: "70%",
            display: "flex",
          }}
        >
          <QueueAnim
            key="queue"
            leaveReverse
            className={styles["statistic-container"]}
            onEnd={(e) => {
              console.log("end", e);
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
              <div key={index} className={styles["statistic"]}>
                <div className={styles["statistic-button"]}>
                {/* <Button onClick={handleClick} style={{ marginTop: 16 }}> */}
                  <TweenOne
                    animation={{
                      Children: {
                        value: values[index],
                        // value: value,
                        floatLength: 0,
                        formatMoney: true,
                      },
                      delay: 0,
                      duration: 700,
                    }}
                    className={styles.numberdisplay}
                  />
                {/* </Button> */}
                </div>
                <div className={styles["statistic-label"]}>
                  <div className={styles["statistic-label-text"]}>
                  {buttonContent.label}
                  </div>
                  <div className={styles["statistic-label-link"]}>
                  {buttonContent.labelLink && (
                    <Link href={buttonContent.labelLink}>
                      {buttonContent.labelLinkText}
                    </Link>  
                  )}
                  </div>
                </div>
              </div>
            ))}
          </QueueAnim>
        </OverPack>
        <OverPack
          style={{
            overflow: "hidden",
            height: "100vh",
            width: "100%",
            display: "flex",
          }}
        >
          <QueueAnim
            key="queue"
            leaveReverse
            className={styles["statistic-container"]}
          >
            <div key="a" className={styles["statistic"]}>
              <div className={styles.statisticlabel}>Total Value</div>
              <Button onClick={handleClick} style={{ marginTop: 16 }}>
                <TweenOne
                  animation={{
                    Children: {
                      value: value,
                      floatLength: 0,
                      formatMoney: true,
                    },
                  }}
                  className={styles.numberdisplay}
                />
              </Button>
            </div>
            <div key="b" className={styles["statistic"]}>
              <Button onClick={handleClick} style={{ marginTop: 16 }}>
                <TweenOne
                  animation={{
                    Children: {
                      value: value,
                      floatLength: 0,
                      formatMoney: true,
                    },
                  }}
                  className={styles.numberdisplay}
                />
              </Button>
            </div>
          </QueueAnim>
        </OverPack>
      </Carousel>
    </div>
  );
};

export default Statistics;
