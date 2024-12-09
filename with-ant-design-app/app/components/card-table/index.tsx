"use client";

import {
  EditOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import {
  Row,
  Col,
  Space,
  Pagination,
} from "antd";

import styles from "./Groups.module.css";
import { useRouter, useSearchParams } from "next/navigation";
import NewGroupModal from "./new-card-modal";
import { useEffect, useRef, useState } from "react";
import { BiBuildings, BiLeftArrow } from "react-icons/bi";
import { useSession } from "next-auth/react";
import { StyledTextField } from "../my-profile/my-profile-form/MyProfileForm";
import { InputAdornment } from "@mui/material";
import { Search } from "@mui/icons-material";
import { FaArrowLeft } from "react-icons/fa6";
import SearchInput from "../search-input/SearchInput";

const CardTable = ({
  columnCount,
  groups = [],
  deleteGroup = () => { },
  createGroup = () => { },
  updateGroup = () => { },
}: {
  columnCount: number;
  groups: any[];
  deleteGroup: (id: string) => void;
  createGroup: (groupName: string) => void;
  updateGroup: (
    groupId: string,
    newGroupName: string
  ) => void;
}): JSX.Element => {
  const router = useRouter();

  const searchParams = useSearchParams();
  // const [searchQuery, setSearchQuery] = useState(searchParams.get('q') || '');

  const { status } = useSession();

  const [newGroupModalVisible, setNewGroupModalVisible] =
    useState(false);

  const [editGroupId, setEditGroupId] = useState(null);
  const [editGroupName, setEditGroupName] = useState("");

  const rowGutter: [number, number] = [8, 32];
  const colSpan: number = 24 / columnCount;

  const handleAddButtonClick = () => {
    setNewGroupModalVisible(true); // Open modal on add button click
  };

  const cardRef = useRef(null);

  if (status === "loading") {
    return <div></div>;
  }

  return (
    <>
      <NewGroupModal
        open={newGroupModalVisible}
        setOpen={setNewGroupModalVisible}
        isEditing={editGroupId !== null}
        groupName={editGroupName}
        groupId={editGroupId ?? ""}
        setGroupName={setEditGroupName}
        addGroup={createGroup}
        deleteGroup={deleteGroup}
        onSubmit={(groupName: string) => {
          if (editGroupId !== null) {
            updateGroup(editGroupId, groupName);
            setEditGroupId(null);
          } else {
            createGroup(groupName);
          }
          setNewGroupModalVisible(false);
        }}
      />
      <SearchInput 
        placeholder="Meklēt grupu pēc tās nosaukuma..." 
        
        onChange={(e: any) => {
          const value = e.target.value;
          // setSearchQuery(value);

          const params = new URLSearchParams(searchParams);
          if (value) {
            params.set('groupName', value);
          } else {
            params.delete('groupName');
          }

          router.replace(`?${params.toString()}`, { scroll: false });
        }} 
      />
      <div
        style={{
          display: "flex",
          justifyContent: "center",
        }}
        className={styles["groups-page"]}
      >

        <Row
          gutter={rowGutter}
          style={{
            width: columnCount < 4 ? "90%" : "100%",
            marginTop: 60,
            marginBottom: 60,
            // justifyContent: "space-between",
          }}
        >
          <Col
            span={colSpan}
            style={{
              display: "flex",
              justifyContent: "space-evenly",
            }}
          >
            <div
              className={`${styles["card"]} ${styles["card-add"]}`}
              onClick={handleAddButtonClick}
            >
              <div className={styles["content"]}>
                <div
                  className={styles["card-content-image"]}
                >
                  <PlusOutlined
                    width={200}
                    height={200}
                  />
                </div>
                <span
                        className={
                          styles[
                          "card-content-title-text-name"
                          ]
                        }
                      >
                  {"Pievienot jaunu grupu"}
                </span>
              </div>
            </div>
          </Col>
          {groups.map((group, index) => (
            <Col
              span={colSpan}
              key={index}
              style={{
                display: "flex",
                justifyContent: "space-evenly",
              }}
            >
              <div
                className={styles["card"]}
                onClick={() => {
                  router.push(`/groups/${group.id}`);
                }}
                ref={cardRef}
              >
                <div
                  className={styles["edit-group-dropdown"]}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditGroupId(group.id);
                    setEditGroupName(group.name);
                    setNewGroupModalVisible(true);
                  }}
                >
                  <Space>
                    <EditOutlined />
                  </Space>
                </div>
                <div className={styles["content"]}>
                  {/* <Image
                    src={group.imageUrl}
                    alt={""}
                    width={200}
                    height={200}
                  /> */}
                  <div
                    style={{
                      background: `url(${group.imageUrl})`,
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                      width: "280px",
                      height: "380px",
                      borderRadius: "10px",
                    }}
                  ></div>
                  <div
                    className={styles["card-content-title"]}
                  >
                    <div
                      className={
                        styles["card-content-title-text"]
                      }
                    >
                      <span
                        className={
                          styles[
                          "card-content-title-text-name"
                          ]
                        }
                      >{group.name}</span>
                      <span
                        className={
                          styles[
                          "card-content-title-object-count"
                          ]
                        }
                      >
                        <span className={styles["card-content-title-object-count-span"]}>
                          {group.residenceCount ?? 0}
                        </span>
                        <BiBuildings />
                      </span>
                    </div>
                    <span className={styles["created-at"]}>
                      Pievienota:{" "}
                      {new Date(
                        group.createdAt
                      ).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </Col>
          ))}
        </Row>
      </div>

      <Pagination
          style={{
            display: "flex",
            justifyContent: "center",
            padding: "1rem",
            // backgroundColor: "var(--background-dark-main-hover)",
            borderRadius: "1rem",
            margin: "1rem 0",
            color: "var(--background-light-secondary)",
            outline: "1px solid var(--background-light-main)",
          }}
          total={85}
          showSizeChanger
          selectPrefixCls="ant-select"
          // showQuickJumper
          showTotal={(total) => `Kopā ${total} ieraksti`}
          
          // itemRender={(current, type, originalElement) => {
          //   if (type === "prev") {
          //     return <FaArrowLeft style={{
          //       fontSize: "1.5rem",
          //       color: "var(--background-light-secondary)",
          //     }} />;
          //   }
          //   if (type === "next") {
          //     return <FaArrowLeft style={{
          //       fontSize: "1.5rem",
          //       color: "var(--background-light-secondary)",
          //       transform: "rotate(180deg)",
          //     }} />;
          //   }
          //   return originalElement;
          // }
          // }
        />

    </>
  );
};

export default CardTable;
