import React from "react";
import styles from "./Sidebar.module.css"; // Create Sidebar.module.css if separate styles needed
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useRouter } from "next/navigation";

interface SidebarProps {
  sidebarItems: {
    id: number;
    object_id?: string;
    icon: JSX.Element;
    label: string;
    item: JSX.Element;
    onClick?: () => void;
  }[];
  activeNavItem: number;
  onNavClick: (id: number | string) => void;
  title?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
  sidebarItems,
  activeNavItem,
  onNavClick,
  title,
}) => {
  const router = useRouter();

  return (
    <div className={styles["left-sidebar"]}>
      <div className={styles["left-sidebar-header"]}>
        <div
          className={styles["left-sidebar-back-arrow"]}
          onClick={() => router.back()}
        >
          <ArrowLeftOutlined />
        </div>
        <div
          className={styles["left-sidebar-header-title"]}
        >
          <span>{title}</span>
        </div>
      </div>
      <div className={styles["left-sidebar-items"]}>
        <div
          className={styles.indicator}
          style={{ top: `${(activeNavItem - 1) * 8}em` }}
        ></div>
        {sidebarItems.map((item) => (
          <div
            key={item.id}
            className={`${styles["left-sidebar-item"]} ${
              activeNavItem === item.id ? styles.active : ""
            }`}
            onClick={() =>
              onNavClick(
                item.object_id ? item.object_id : item.id
              )
            }
          >
            {item.icon}
            <span
              className={styles["left-sidebar-item-label"]}
            >
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
