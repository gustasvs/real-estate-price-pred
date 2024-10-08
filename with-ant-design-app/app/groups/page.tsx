"use client";

import React from "react";
import CardTable from "../components/card-table";

import styles from "../components/groups/GroupsPage.module.css";
import GenericLayout from "../components/generic-page-layout";

import { useRouter } from "next/navigation";

const GroupsPage = () => {

  const router = useRouter();

  const openGroup = (id: number) => {
    console.log(`Open group with id ${id}`);
    router.push(`/groups/${id}`);
  };

  return (
    <GenericLayout>
      {/* <div style={{ display: "flex", justifyContent: "center" }} className={styles["groups-page"]}> */}
        <CardTable columnCount={3} onCardClick={(id: number) => openGroup(id)}/>
      {/* </div> */}
    </GenericLayout>
  );
};

export default GroupsPage;
