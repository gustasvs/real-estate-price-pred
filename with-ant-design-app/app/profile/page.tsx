"use client";

import React, { useState } from "react";
import { Divider } from "antd";
import { PageHeader } from "@ant-design/pro-components";
import { HeartOutlined, UploadOutlined, UserOutlined } from "@ant-design/icons";
import styles from "./Profile.module.css";
import GenericLayout from "../components/generic-page-layout";
import { IoSettingsOutline } from "react-icons/io5";
import MyProfileForm from "../components/my-profile/my-profile-form/MyProfileForm";

import { useRouter, useSearchParams } from "next/navigation";


const sidebarItems = [
  {
    id: 1,
    icon: <UserOutlined />,
    label: "Lietotāja informācija",
    item: <MyProfileForm />,
  },
  {
    id: 2,
    icon: <HeartOutlined />,
    label: "Atzīmetas dzīvesvietas",
    item: <div>Atzīmetas dzīvesvietas</div>,
  },
  {
    id: 3,
    icon: <IoSettingsOutline />,
    label: "Iestatījumi",
    item: <div>Iestatījumi</div>,
  },
];

const UserProfilePage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const initialNavItem = Number(searchParams.get("activeNavItem")) || 1;
  const [activeNavItem, setActiveNavItem] = useState(initialNavItem);
  const [prevActiveNavItem, setPrevActiveNavItem] = useState(initialNavItem);


  const handleNavClick = (newActiveNavItem: number) => {
    setPrevActiveNavItem(activeNavItem);
    setActiveNavItem(newActiveNavItem);
  };

  console.log(activeNavItem, prevActiveNavItem);
  

  return (
    <GenericLayout>
      <div className={styles["profile-page-container"]}>
        <div className={styles["left-sidebar"]}>

          <div className={styles["left-sidebar-header"]}>
            <div className={styles["left-sidebar-header-title"]}>
              <span>Mans profils</span>
            </div>
          </div>

          {/* Sidebar items */}
          <div className={styles["left-sidebar-items"]}>
            {/* Indicator */}
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
                onClick={() => handleNavClick(item.id)}
              >
                {item.icon}
                <span className={styles["left-sidebar-item-label"]}>
                  {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className={`${styles['main-content']} ${activeNavItem > prevActiveNavItem ? 'slide-down' : 'slide-up'}`}>
          {sidebarItems &&
            sidebarItems.find((item) => item.id === activeNavItem)?.item}
        </div>
      </div>
    </GenericLayout>
  );
};

export default UserProfilePage;
