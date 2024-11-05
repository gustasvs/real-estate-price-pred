"use client";

import React, { Suspense, useEffect, useState } from "react";
import { Divider } from "antd";
import { HeartOutlined, UploadOutlined, UserOutlined } from "@ant-design/icons";
import styles from "./Profile.module.css";
import GenericLayout from "../components/generic-page-layout";
import { IoSettingsOutline } from "react-icons/io5";
import MyProfileForm from "../components/my-profile/my-profile-form/MyProfileForm";

import { useRouter, useSearchParams } from "next/navigation";
import Sidebar from "../components/navigation/sidebar/Sidebar";
import MyFavouritedObjects from "../components/my-profile/my-profile-form/my-favourited-objects/MyFavouritedObjects";
import PageHeader from "../components/generic-page-layout/page-header/PageHeader";


const sidebarItems = [
  {
    id: 0,
    icon: <UserOutlined />,
    label: "Lietotāja informācija",
    item: <MyProfileForm />,
  },
  {
    id: 1,
    icon: <HeartOutlined />,
    label: "Atzīmetas dzīvesvietas",
    item: <MyFavouritedObjects />
  },
  {
    id: 2,
    icon: <IoSettingsOutline />,
    label: "Iestatījumi",
    item: <div>Settings</div>,
  },
];

const UserProfilePage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const initialNavItem = Number(searchParams.get("page")) || 0;
  // const [activeNavItem, setActiveNavItem] = useState(initialNavItem);
  const activeNavItem = initialNavItem;
  const [prevActiveNavItem, setPrevActiveNavItem] = useState(initialNavItem);



  // const handleNavClick = (newActiveNavItem: number | string) => {
  //   setPrevActiveNavItem(activeNavItem);
  //   // router.push(`/profile?page=${newActiveNavItem}`, { shallow: true } as any);
  //   window.history.pushState(null, "", `?page=${newActiveNavItem}`);
  //   // setActiveNavItem(Number(newActiveNavItem));
  // };

  // TODO slowLoading for sidebar items for group page
  // TODO this means that the sidebar items will be loaded only when clicked

  return (
    <GenericLayout>
      <PageHeader
        title="Mans profils"
        breadcrumbItems={[
          {
            label: "Mans profils",
            path: "/profile",
          },
          {
            label: sidebarItems.find((item) => item.id === activeNavItem)?.label || "Lietotāja informācija",
            path: `/profile?page=${activeNavItem}`,
          }
          
        ]}
      />

      <div className={styles["profile-page-container"]}>

          {/* <Sidebar sidebarItems={sidebarItems} activeNavItem={activeNavItem} onNavClick={handleNavClick} title="Mans profils" /> */}

        <div className={`${styles['main-content']} ${activeNavItem > prevActiveNavItem ? 'slide-down' : 'slide-up'}`}>
          {/* <Suspense fallback={<div style={{ height: "60vh", backgroundColor: "white",
          width: "100%", display: "flex", justifyContent: "center", alignItems: "center" }} />}> */}
          {sidebarItems &&
            sidebarItems.find((item) => item.id === activeNavItem)?.item}
          {/* </Suspense> */}
        </div>
      </div>
    </GenericLayout>
  );
};

export default UserProfilePage;
