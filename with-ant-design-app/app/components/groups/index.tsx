import { PlusOutlined } from "@ant-design/icons";
import { Card, Row, Col } from "antd";
import Image from "next/image";

import styles from "./Groups.module.css";

const groups = [
  { id: 1, name: "Group 1", imageUrl: "/path/to/image1.jpg" },
  { id: 2, name: "Group 2", imageUrl: "/path/to/image2.jpg" },
  { id: 3, name: "Group 3", imageUrl: "/path/to/image3.jpg" },
  { id: 4, name: "Group 4", imageUrl: "/path/to/image4.jpg" },
];

const Groups = ({}: {}): JSX.Element => {
  return (
    <>
      <Row
        gutter={[16, 16]}
        style={{
          width: "60%",
          marginTop: 200,
        }}
      >
        {groups.map((group, index) => (
          <Col
            span={8}
            key={index}
            style={{
              display: "flex",
              justifyContent: "space-evenly",
            }}
          >
            <div className={styles["card"]}>
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
        <Col span={8}>
          <div className={`${styles["card"]} ${styles["card-add"]}`}>
            <div className={styles["content"]}>
              <PlusOutlined
                style={{
                  justifyContent: "center",
                  alignItems: "center",
                    display: "flex",
                  marginTop: "30px",
                  marginBottom: "40px",
                  width: 200,
                  height: 200,
                }}
              />
            </div>
          </div>
        </Col>
      </Row>
    </>
  );
};

export default Groups;
