"use client";

import { PlusOutlined } from "@ant-design/icons";
import { Card, Row, Col, Button, Descriptions, Divider } from "antd";
import Image from "next/image";

import styles from "./MasonryTable.module.css";
import { useRouter } from "next/navigation";

import Layout from "react-masonry-list";
import { PageHeader } from "@ant-design/pro-components";

const MasonryTable = ({
  columnCount,
  onCardClick = () => {},
  objects,
  setObjects,
}: {
  columnCount: number;
  onCardClick?: (id: number) => void;
  objects: any[];
  setObjects: (objects: any[]) => void;
}): JSX.Element => {
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
      setObjects(randomOrder);
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

  const routes = [
    {
      path: 'index',
      breadcrumbName: 'First-level Menu',
    },
    {
      path: 'first',
      breadcrumbName: 'Second-level Menu',
    },
    {
      path: 'second',
      breadcrumbName: 'Third-level Menu',
    },
  ];

  return (
    <div
    className={styles["masonry-table-container"]}
    >
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
          <Button onClick={animateAndSort} key="3">Kārtot pēc pievienošanas laika</Button>,
          <Button onClick={animateAndSort} key="2">Kārtot pēc cenas</Button>,
          <Button onClick={animateAndSort} type="primary">
            <span>Kārtot</span>
          </Button>
        ]}
        
      >
      </PageHeader>
      <Divider />
      <div className={styles["masonry-table"]}>
        {objects.map((item) => (
          <div
            key={item.id}
            id={`item-${item.id}`}
            className={styles["content"]}
            onClick={() => onCardClick(item.id)}
          >
            <div
              className={styles["content-image"]}
              style={{
                background: `#${Math.floor(Math.random() * 16777215).toString(
                  16
                )}`,
                height: `${item.height}px`,
              }}
            ></div>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: 10,
            }}>
            <span style={{ color: "#ffffff", marginLeft: 20, fontSize: 22}}>{item.name}</span>
              </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MasonryTable;
