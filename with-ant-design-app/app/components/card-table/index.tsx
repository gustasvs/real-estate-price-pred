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
  Button,
} from "antd";

import styles from "./Groups.module.css";
import { useRouter, useSearchParams } from "next/navigation";
import NewGroupModal from "./new-card-modal";
import { useEffect, useRef, useState } from "react";
import { BiBuildings, BiLeftArrow } from "react-icons/bi";
import { useSession } from "next-auth/react";
import SearchInput from "../search-input/SearchInput";
import { getPlural } from "../masonry-table";
import CustomPagination from "../pagination/CustomPagination";

const CardTable = ({
  columnCount,
  groups = [],
  deleteGroup = () => { },
  createGroup = () => { },
  updateGroup = () => { },
  total = 0,
}: {
  columnCount: number;
  groups: any[];
  total: number;
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

  const [loading, setLoading] = useState(false);

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

  useEffect(() => {
    setLoading(false);
  }, [groups]);

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
      <div className={styles["card-table-header"]}>
        <SearchInput
          placeholder="Meklēt grupu pēc tās nosaukuma..."

          defaultValue={searchParams.get('groupName') || ''}
          style={{ marginTop: 0, marginBottom: 0, width: "100%" }}

          onChange={(e: any) => {

            const value = e.target.value;
            // setSearchQuery(value);

            const params = new URLSearchParams(searchParams);
            if (value) {
              params.set('groupName', value);
            } else {
              params.delete('groupName');
            }

            params.set('page', '1');

            router.replace(`?${params.toString()}`, { scroll: false });

            setLoading(true);
          }}
        />
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleAddButtonClick}
          className={styles["add-group-button"]}
        >
          Pievienot jaunu grupu
        </Button>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          filter: `${loading ? "blur(2px)" : "none"}`,
          transform: `${loading ? "scale(0.97)" : "scale(1)"}`,
          transition: "all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        }}
        className={`${styles["groups-page"]} ${groups.length === 0 ? styles["groups-page-empty"] : ""}`}
      >
        {/* {loading && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                width: "100%",
              }}
            >
              <div
                className={styles["loading-groups"]}
              >
              </div>
              <span
                style={{
                  color: "var(--background-light-main)",
                  fontSize: "1.5rem",
                  margin: "1rem",
                }}
              >
                Meklē grupas...
              </span>
            </div>

          )} */}
        {groups.length === 0 ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              width: "100%",
            }}
          >
            <div
              className={styles["empty-groups"]}
            >
            </div>
            <span
              style={{
                color: "var(--background-light-main)",
                fontSize: "1.5rem",
                margin: "1rem",
              }}
            >
              Nav atrasta neviena grupa
            </span>
          </div>
        ) :
          (
            <Row
              gutter={rowGutter}
              style={{
                width: columnCount < 4 ? "90%" : "100%",
                marginTop: 60,
                marginBottom: 60,
                // justifyContent: "space-between",
              }}
            >
              {/* <Col
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
                <div
                  className={
                    styles[
                    "card-content-title"
                    ]
                  }
                >
                  <span className={
                    styles[
                    "card-content-title-text-name"
                    ]
                  }>
                    {"Pievienot jaunu grupu"}
                  </span>
                </div>
              </div>
            </div>
          </Col> */}

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
          )
        }
      </div>
      <div className={styles["card-table-footer"]}>

        <CustomPagination
          onChange={(page, pageSize) => {
            const params = new URLSearchParams(searchParams);
            params.set('page', page.toString());
            params.set('pageSize', pageSize.toString());
            router.replace(`?${params.toString()}`, { scroll: false });
          }}
          total={total}
          showTotal={(total) => `Kopā: ${total} ${getPlural(total, "grupa", "grupas")}`}
          pageSizeOptions={["3", "6", "9", "12"]}
          defaultCurrent={parseInt(searchParams.get('page') || '1')}
          defaultPageSize={parseInt(searchParams.get('pageSize') || '6')}
        />
      </div>
    </>
  );
};

export default CardTable;
