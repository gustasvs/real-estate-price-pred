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
import { headers } from "next/headers";


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
    label: "Personalizācija un iestatījumi",
    item: <div>Settings</div>,
  },
];

const UserProfilePage = async () => {
  
  const headersList = headers();
  const query = new URL(headersList.get("referer") || "").searchParams;
  const initialNavItem = Number(query.get("page")) || 0;


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
            label:
              sidebarItems.find((item) => item.id === initialNavItem)?.label ||
              "Lietotāja informācija",
            path: `/profile?page=${initialNavItem}`,
          },
        ]}
      />

      <div className={styles["profile-page-container"]}>

          {/* <Sidebar sidebarItems={sidebarItems} activeNavItem={activeNavItem} onNavClick={handleNavClick} title="Mans profils" /> */}

        <div className={`${styles['main-content']}`}>
          {/* <Suspense fallback={<div style={{ height: "60vh", backgroundColor: "white",
          width: "100%", display: "flex", justifyContent: "center", alignItems: "center" }} />}> */}
          {sidebarItems &&
            sidebarItems.find((item) => item.id === initialNavItem)?.item}
          {/* </Suspense> */}
        </div>
      </div>
    </GenericLayout>
  );
};

export default UserProfilePage;
