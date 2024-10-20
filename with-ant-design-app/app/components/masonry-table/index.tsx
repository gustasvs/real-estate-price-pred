"use client";

import { EditOutlined, PlusOutlined } from "@ant-design/icons";
import { FaShower } from "react-icons/fa6";
import { IoBedOutline } from "react-icons/io5";
import { FaCarSide, FaRuler } from "react-icons/fa";
import { Card, Row, Col, Button, Descriptions, Divider, Grid } from "antd";
import Image from "next/image";

import styles from "./MasonryTable.module.css";
import { useRouter } from "next/navigation";

import Layout from "react-masonry-list";
import { PageHeader } from "@ant-design/pro-components";
import { useState } from "react";

const MasonryTable = ({
  columnCount,
  onCardClick = () => {},
  objects,
  createObject = () => {},
  deleteObject = () => {},
  updateObject = () => {},
  loading = false,
}: {
  columnCount: number;
  onCardClick?: (id: number) => void;
  objects: any[];
  createObject: () => void;
  deleteObject: (id: string) => void;
  updateObject: (
    id: string,
    objectData: { name: string; description: string; pictures: string[] }
  ) => void;
  loading?: boolean;
}): JSX.Element => {
  const [newObjectModalVisible, setNewObjectModalVisible] = useState(false);

  const handleAddButtonClick = () => {
    createObject();
  };

  const animateAndSort = () => {
    objects.forEach((object) => {
      const element = document.getElementById(`item-${object.id}`);
      if (element) {
        const randomX = 10 * (Math.random() > 0.5 ? 1 : -1);
        const randomY = 10 * (Math.random() > 0.5 ? 1 : -1);
        element.style.transition = "all 0.48s cubic-bezier(0.23, 1, 0.32, 1)";
        element.style.opacity = "0.5";
        element.style.transform = `translate(${randomX}px, ${randomY}px)`;
      }
    });

    setTimeout(() => {
      const randomOrder = [...objects];
      randomOrder.sort(() => Math.random() - 0.5);
      // setObjects(randomOrder);
      setTimeout(() => {
        objects.forEach((object) => {
          const element = document.getElementById(`item-${object.id}`);
          if (element) {
            element.style.transition =
              "all 0.48s cubic-bezier(0.23, 1, 0.32, 1)";
            element.style.opacity = "1"; // Fade in
            element.style.transform = "translate(0, 0)";
          }
        });
      }, 10);
    }, 300);
  };

  const distributeItems = (items: any[], pattern: number[]) => {
    let result = [];
    let index = 0;
    let rowIndex = 0;
    while (index < items.length) {
      let row = [];
      let count = pattern[rowIndex % pattern.length];
      for (let i = 0; i < count; i++) {
        if (index < items.length) {
          row.push(items[index]);
          index++;
        }
      }
      result.push(row);
      rowIndex++;
    }
    return result;
  };

  const rowPattern = [2, 4, 3]; // This defines the pattern of rows per column
  const rowItems = distributeItems(objects, rowPattern);

  console.log("rowItems", rowItems);

  const routes = [
    {
      path: "index",
      breadcrumbName: "First-level Menu",
    },
    {
      path: "first",
      breadcrumbName: "Second-level Menu",
    },
    {
      path: "second",
      breadcrumbName: "Third-level Menu",
    },
  ];

  return (
    <div className={styles["masonry-table-container"]}>
      <PageHeader
        ghost={false}
        onBack={() => window.history.back()}
        title="Grupas objekti"
        subTitle="Šeit var redzēt visus grupas objektus"
        breadcrumb={{ items: routes }}
        breadcrumbRender={(props, originBreadcrumb) => {
          return originBreadcrumb;
        }}
        className={styles["site-page-header"]}
        extra={[
          <Button onClick={animateAndSort} key="3">
            Kārtot pēc pievienošanas laika
          </Button>,
          <Button onClick={animateAndSort} key="2">
            Kārtot pēc cenas
          </Button>,
          <Button onClick={animateAndSort} type="primary">
            <span>Kārtot</span>
          </Button>,
        ]}
      ></PageHeader>
      <Divider />
      {/* <div className={styles["masonry-table"]}> */}
      <Row
        gutter={{ xs: 8, sm: 16, md: 24, lg: 32 }}
        style={{ width: "90%", margin: "0 auto" }}
      >
        {rowItems.map((itemsInRow, rowIndex) => (
          <Row
            key={`row-${rowIndex}`}
            style={{ width: "100%" }}
            gutter={[42, 32]}
          >
            {itemsInRow.map((item, itemIndex) => (
              <Col
                className={styles["content-wrapper"]}
                span={Math.min(24 / itemsInRow.length, 12)}
                key={item.id}
              >
                <div
                  id={`item-${item.id}`}
                  className={styles["content"]}
                  onClick={() => onCardClick(item.id)}
                  key={item.id}
                  style={{
                    marginBottom: 20, // Maintain bottom margin for spacing
                  }}
                >
                  <div className={styles["content-image"]}>
                    <img
                      src={
                        item.pictures[0]?.startsWith("data:image")
                          ? item.pictures[0]
                          : `data:image/png;base64,${item.pictures[0]}`
                      }
                      alt="content"
                      style={{
                        height: "100%", // Ensure image matches the container's height
                        width: "auto", // Maintain aspect ratio and scale the width accordingly
                        display: "block", // Remove any inline image spacing issues
                      }}
                    />
                  </div>
                  {/* other images */}
                  <div className={styles["content-images-other"]}>
                  {item.pictures.length > 1 && (
                      <img
                        src={
                          item.pictures[1]?.startsWith("data:image")
                            ? item.pictures[1]
                            : `data:image/png;base64,${item.pictures[1]}`
                        }
                        alt="content"
                        style={{
                          height: "100%", // Ensure image matches the container's height
                          width: "auto", // Maintain aspect ratio and scale the width accordingly
                          display: "block", // Remove any inline image spacing issues
                        }}
                      />
                  )}
                  </div>
                    

                  <div
                    className={styles["content-title-wrapper"]}
                  >
                    <span
                      className={styles["content-title-name"]}
                    >
                      {item.name}
                    </span>
                  </div>

                  <div className={styles["content-description-wrapper"]}>
                    <div className={styles["content-description"]}>
                      {/* Header */}
                      <div className={styles["content-description-header"]}>
                        <div
                          className={styles["content-description-header-top"]}
                        >
                          <span
                            className={
                              styles["content-description-header-name"]
                            }
                          >
                            {item.name}
                          </span>
                          <span
                            className={
                              styles["content-description-header-price"]
                            }
                          >
                            {item.price ?? "Nav norādīta cena"}
                          </span>
                        </div>
                        <div
                          className={
                            styles["content-description-header-bottom"]
                          }
                        >
                          <span
                            className={
                              styles["content-description-header-date"]
                            }
                          >
                            Pievienots{" "}
                            {new Date(item.createdAt).toLocaleDateString()}
                          </span>
                          <span
                            className={
                              styles[
                                "content-description-header-price-prediction"
                              ]
                            }
                          >
                            {item.pricePrediction ??
                              "Nav norādīta cenu prognoze"}
                          </span>
                        </div>
                      </div>

                      {/* Details list */}
                      <div className={styles["content-description-list"]}>
                        <li className={styles["content-description-list-item"]}>
                          <span
                            className={
                              styles["content-description-list-item-title"]
                            }
                          >
                            Pieraksti:
                          </span>
                          <span
                            className={
                              styles["content-description-list-item-value"]
                            }
                          >
                            {item.description}
                          </span>
                        </li>
                        <li className={styles["content-description-list-item"]}>
                          <span
                            className={
                              styles["content-description-list-item-title"]
                            }
                          >
                            Adrese:
                          </span>
                          <span
                            className={
                              styles["content-description-list-item-value"]
                            }
                          >
                            {item.address ?? "Rīga, Jaunā iela 1 - 22"}
                          </span>
                        </li>
                        <li className={styles["content-description-list-item"]}>
                          <span
                            className={
                              styles["content-description-list-item-title"]
                            }
                          >
                            Platība:
                          </span>
                          <span
                            className={
                              styles["content-description-list-item-value"]
                            }
                          >
                            {`${item.area ?? 100} m²`}
                          </span>
                        </li>
                      </div>

                      {/* Footer */}
                      <div className={styles["content-description-footer"]}>
                        <div
                          className={
                            styles["content-description-house-details"]
                          }
                        >
                          <div
                            className={styles["content-description-bed-counts"]}
                          >
                            <IoBedOutline />
                            <span>{item.bedroomCount ?? "3 Gultas"}</span>
                          </div>
                          <div
                            className={
                              styles["content-description-bath-counts"]
                            }
                          >
                            <FaShower />
                            <span>{item.bathroomCount ?? "2 Dušas"}</span>
                          </div>
                          <div
                            className={styles["content-description-house-area"]}
                          >
                            <FaCarSide />
                            <span>{`${item.parkingCount ?? "2"} Stāvvietas`}</span>
                          </div>
                        </div>

                        <div className={styles["content-description-actions"]}>
                          <Button
                            type="primary"
                            onClick={() => updateObject(item.id, item)}
                            className={`${styles["content-description-action"]} ${styles["content-description-action-edit"]}`}
                          >
                            <EditOutlined />
                            Rediģēt
                          </Button>
                          <Button
                            type="primary"
                            onClick={() => deleteObject(item.id)}
                            className={`${styles["content-description-action"]} ${styles["content-description-action-delete"]}`}
                          >
                            Dzēst
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Col>
            ))}
          </Row>
        ))}
      </Row>

      {/* <div
          className={`${styles["content"]} ${styles["content-add"]}`}
          onClick={handleAddButtonClick}
        >
          <div className={styles["content-add-image"]}>
            <PlusOutlined style={{ fontSize: "48px", color: "#fff" }} />
          </div>
          <span className={styles["content-add-title"]}>
            Pievienot jaunu objektu
          </span>
        </div> */}
    </div>
  );
};

export default MasonryTable;
