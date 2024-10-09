import { CloseOutlined, PlusOutlined } from "@ant-design/icons";
import { Card, Row, Col } from "antd";
import Image from "next/image";

import styles from "./Groups.module.css";
import { useRouter } from "next/navigation";


const CardTable = ({
  columnCount,
  onCardClick = () => {},
  groups,
  setGroups,
}: {
  columnCount: number;
  onCardClick?: (id: number) => void;
  groups: any[];
  setGroups: (groups: any[]) => void;
}): JSX.Element => {
  const router = useRouter();

  const rowGutter: [number, number] = [16, 16];
  const colSpan: number = 24 / columnCount;

  const addNewGroup = () => {
    const newGroups = [...groups];
    newGroups.push({ id: groups.length + 1, name: "Jauna grupa", imageUrl: "/path/to/image.jpg" });
    setGroups(newGroups);
  }

  const deleteGroup = (id: number) => {
    const newGroups = groups.filter((group) => group.id !== id);
    setGroups(newGroups);
  }


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
                  <div className={styles["delete-group-button"]} onClick={(e) => {
                      e.stopPropagation();
                      deleteGroup(group.id);
                  }}>
                    <CloseOutlined />
                  </div>
                  {/* <Image
                    src={group.imageUrl}
                    alt={""}
                    width={200}
                    height={200}
                  /> */}
                  <div style={{
                      background: `url(${group.imageUrl})`,
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                      width: "200px",
                      height: "200px",
                      borderRadius: "10px",
                  }}>

                  </div>
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
            <div className={`${styles["card"]} ${styles["card-add"]}`} onClick={() => {
                addNewGroup();
            }}>
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
