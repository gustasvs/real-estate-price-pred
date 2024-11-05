import React from "react";
import styles from "./Sidebar.module.css"; // Create Sidebar.module.css if separate styles needed
import {
  ArrowLeftOutlined,
  GitlabOutlined,
} from "@ant-design/icons";
import { useRouter } from "next/navigation";
import Logo from "../navbar/Logo";
import { BiBookAdd } from "react-icons/bi";

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
  slowLoading?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  sidebarItems,
  activeNavItem,
  onNavClick,
  title,
  slowLoading = false,
}) => {
  const router = useRouter();

  return (
    <div className={styles["left-sidebar"]}>
      <div className={styles["left-sidebar-company-logo"]}>
        <span
          className={styles["left-sidebar-company-title"]}
        >
          Icn
        </span>
        <Logo />
      </div>

      {/* <div className={styles["left-sidebar-header"]}>
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
      </div> */}
      <div className={styles["left-sidebar-items"]}>
        <div
          className={styles.indicator}
          style={{ top: `${activeNavItem * 8}em` }}
        />

        <div className={styles["left-sidebar-group"]}>
          <span
            className={styles["left-sidebar-group-title"]}
          >
            Sakums
          </span>

          <div
            key="home"
            className={styles["left-sidebar-item"]}
          >
            <GitlabOutlined />
            <span
              className={styles["left-sidebar-item-label"]}
            >
              Sākums
            </span>
          </div>
        </div>

        <

        <div
          key="my-groups"
          className={styles["left-sidebar-item"]}
        >
          <BiBookAdd />
          <span
            className={styles["left-sidebar-item-label"]}
          >
            Manas grupas
          </span>
        </div>
        {sidebarItems?.map((item) => (
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
