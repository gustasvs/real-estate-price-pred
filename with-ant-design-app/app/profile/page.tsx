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
import Sidebar from "../components/navigation/sidebar/Sidebar";


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


  const handleNavClick = (newActiveNavItem: number | string) => {
    setPrevActiveNavItem(activeNavItem);
    setActiveNavItem(Number(newActiveNavItem));
  };

  return (
    <GenericLayout>
      <div className={styles["profile-page-container"]}>

          <Sidebar sidebarItems={sidebarItems} activeNavItem={activeNavItem} onNavClick={handleNavClick} title="Mans profils" />

        <div className={`${styles['main-content']} ${activeNavItem > prevActiveNavItem ? 'slide-down' : 'slide-up'}`}>
          {sidebarItems &&
            sidebarItems.find((item) => item.id === activeNavItem)?.item}
        </div>
      </div>
    </GenericLayout>
  );
};

export default UserProfilePage;
