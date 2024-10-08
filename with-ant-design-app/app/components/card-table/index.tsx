import { PlusOutlined } from "@ant-design/icons";
import { Card, Row, Col } from "antd";
import Image from "next/image";

import styles from "./Groups.module.css";
import { useRouter } from "next/navigation";

const groups = [
  { id: 1, name: "Group 1", imageUrl: "/path/to/image1.jpg" },
  { id: 2, name: "Group 2", imageUrl: "/path/to/image2.jpg" },
  { id: 3, name: "Group 3", imageUrl: "/path/to/image3.jpg" },
  { id: 4, name: "Group 4", imageUrl: "/path/to/image4.jpg" },
  // { id: 5, name: "Group 5", imageUrl: "/path/to/image5.jpg" },
];

const CardTable = ({
  columnCount,
  onCardClick = () => {},
}: {
  columnCount: number;
  onCardClick?: (id: number) => void;
}): JSX.Element => {
  const router = useRouter();

  const rowGutter: [number, number] = [16, 16];
  const colSpan: number = 24 / columnCount;

  return (
    <>
      <div
        style={{ display: "flex", justifyContent: "center" }}
        className={styles["groups-page"]}
      >
        <Row
          gutter={rowGutter}
          style={{
            width: columnCount < 4 ? "70%" : "80%",
            marginTop: 60,
            marginBottom: 60,
            // justifyContent: "space-between",
          }}
        >
          {groups.map((group, index) => (
            <Col
              span={colSpan}
              key={index}
              style={{
                display: "flex",
                justifyContent: "space-evenly",
              }}
            >
              <div
                className={styles["card"]}
                onClick={() => {
                  onCardClick(group.id);
                }}
              >
                <div className={styles["content"]}>
                  <Image
                    src={group.imageUrl}
                    alt={group.name}
                    width={200}
                    height={200}
                  />
                  <h4 style={{ color: "#ffffff" }}>{group.name}</h4>
                </div>
              </div>
            </Col>
          ))}
          <Col
            span={colSpan}
            style={{
              display: "flex",
              justifyContent: "space-evenly",
            }}
          >
            <div className={`${styles["card"]} ${styles["card-add"]}`}>
              <div className={styles["content"]}>
                <div className={styles["card-content-image"]}>
                  <PlusOutlined
                    style={
                      {
                        //   display: "flex",
                        // marginTop: "30px",
                        // marginBottom: "40px",
                      }
                    }
                    width={200}
                    height={200}
                  />
                </div>
                <h4 style={{ color: "#ffffff" }}>{"Pievienot jaunu"}</h4>
              </div>
            </div>
          </Col>
        </Row>
      </div>
    </>
  );
};

export default CardTable;
