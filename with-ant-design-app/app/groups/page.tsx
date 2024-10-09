"use client";

import React, { useState } from "react";
import CardTable from "../components/card-table";

// import styles from "../components/groups/GroupsPage.module.css";
import GenericLayout from "../components/generic-page-layout";

import { useRouter } from "next/navigation";
import { Divider } from "antd";
import { PageHeader } from "@ant-design/pro-components";

const GroupsPage = () => {

  const router = useRouter();

  const navigateToGroup = (id: number) => {
    console.log(`Open group with id ${id}`);
    router.push(`/groups/${id}`);
  };

  const [groups, setGroups] = useState([
    { id: 1, name: "Rīgas dzīvokļi", imageUrl: "/path/to/image1.jpg" },
    { id: 2, name: "Cēsu dzīvokļi", imageUrl: "/path/to/image2.jpg" },
    { id: 3, name: "Mājas pie Jūrmalas", imageUrl: "/path/to/image3.jpg" },
    { id: 4, name: "Dzīvokļi S. Zvirbulim", imageUrl: "/path/to/image4.jpg" },
    // { id: 5, name: "Group 5", imageUrl: "/path/to/image5.jpg" },
  ]);

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
    <GenericLayout>
      {/* <div style={{ display: "flex", justifyContent: "center" }} className={styles["groups-page"]}> */}
      <PageHeader
        ghost={false}
        onBack={() => window.history.back()}
        title="Manas grupas"
        subTitle="Šeit var redzēt visas manas grupas"
        breadcrumb={{ routes }}
        // breadcrumbRender={(props, originBreadcrumb) => {
        //   return originBreadcrumb;

        // }}
      >
      </PageHeader>
      <Divider />
        <CardTable columnCount={3} onCardClick={(id: number) => navigateToGroup(id)} groups={groups} setGroups={setGroups}/>
      {/* </div> */}
    </GenericLayout>
  );
};

export default GroupsPage;
