import { CloseOutlined, PlusOutlined } from "@ant-design/icons";
import { Card, Row, Col, Popover, Popconfirm } from "antd";
import Image from "next/image";

import styles from "./Groups.module.css";
import { useRouter } from "next/navigation";
import NewGroupModal from "./new-card-modal";
import { useState } from "react";


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

  const [newGroupModalVisible, setNewGroupModalVisible] = useState(false);


  const rowGutter: [number, number] = [16, 16];
  const colSpan: number = 24 / columnCount;

  const addNewGroup = (groupName: string) => {
    const newGroups = [...groups, { id: groups.length + 1, name: groupName, imageUrl: "/path/to/image.jpg" }];
    setGroups(newGroups);
  };

  const deleteGroup = (id: number) => {
    const newGroups = groups.filter((group) => group.id !== id);
    setGroups(newGroups);
  }

  const handleAddButtonClick = () => {
    setNewGroupModalVisible(true); // Open modal on add button click
  };


  return (
    <>
      <NewGroupModal 
        open={newGroupModalVisible} 
        setOpen={setNewGroupModalVisible} 
        addGroup={addNewGroup} 
      />
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
                  <Popconfirm
                    title="Vai tiešām vēlaties dzēst šo grupu?"
                    onConfirm={(e) => {
                      e.stopPropagation();
                      deleteGroup(group.id);
                    }}
                    onCancel={(e) => {
                      e.stopPropagation();
                    }}
                    okText="Jā"
                    cancelText="Nē"
                    okButtonProps={{ className: styles["delete-group-button-popover-ok"] }}
                    cancelButtonProps={{ className: styles["delete-group-button-popover-cancel"] }}
                  >
                  <div className={styles["delete-group-button"]} onClick={(e) => {
                      e.stopPropagation();
                  }}>
                    <CloseOutlined />
                  </div>
                  </Popconfirm >
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
                      height: "280px",
                      borderRadius: "10px",
                  }}>

                  </div>
                  <span className={styles["card-content-title"]}>
                    {group.name}
                  </span>
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
            <div className={`${styles["card"]} ${styles["card-add"]}`} onClick={handleAddButtonClick}>
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
