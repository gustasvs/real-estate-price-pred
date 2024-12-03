"use client";

import {
  EditOutlined,
  HeartFilled,
  HeartOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import QuestionMarkIcon from '@mui/icons-material/QuestionMark';
import {
  FaChevronDown,
  FaChevronUp,
  FaShower,
} from "react-icons/fa6";
import { IoBedOutline } from "react-icons/io5";
import { FaCarSide } from "react-icons/fa";
import {
  Row,
  Col,
  Divider,
} from "antd";

import { IoMdArrowRoundUp } from "react-icons/io";


import styles from "./MasonryTable.module.css";
import { useRouter } from "next/navigation";

import { useState } from "react";

import {
  getObjects as getObjectsApi,
  createObject as createObjectApi,
  deleteObject as deleteObjectApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";
import { useSession } from "next-auth/react";
import { IconButton, Tooltip } from "@mui/material";

const MasonryTable = ({
  group_id,
  columnCount,
  objects,
  loading = false,
  showNavigateToGroup = false,
  revalidateDataFunction = () => { },
}: {
  group_id: string;
  columnCount: number;
  objects: any[];
  loading?: boolean;
  showNavigateToGroup?: boolean;
  revalidateDataFunction?: () => void;
}): JSX.Element => {
  const router = useRouter();

  const { data: session, status, update } = useSession();

  const deleteObject = async (id: string) => {
    const result = await deleteObjectApi(id);
    // await fetchObjects();
  };

  const onCardFavorite = async (id: string) => {
    const object = objects.find(
      (object) => object.id === id
    );
    if (!object) {
      return;
    }
    const updatedObject = await updateObjectApi(id, {
      favourite: !object.favourite,
    });

    revalidateDataFunction();
    // await fetchObjects();
  };

  const [openedDescriptions, setOpenedDescriptions] =
    useState<{ [key: number]: boolean }>({});

  const toggleDescription = (id: number) => {
    setOpenedDescriptions((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const handleAddButtonClick = () => {
    if (group_id === "favourites") {
      router.push(`/groups/${objects[0].groupId}/new`);
    } else {
      router.push(`/groups/${group_id}/new`);
    }
  };

  const animateAndSort = () => {
    objects.forEach((object) => {
      const element = document.getElementById(
        `item-${object.id}`
      );
      if (element) {
        const randomX = 10 * (Math.random() > 0.5 ? 1 : -1);
        const randomY = 10 * (Math.random() > 0.5 ? 1 : -1);
        element.style.transition =
          "all 0.48s cubic-bezier(0.23, 1, 0.32, 1)";
        element.style.opacity = "0.5";
        element.style.transform = `translate(${randomX}px, ${randomY}px)`;
      }
    });

    setTimeout(() => {
      const randomOrder = [...objects];
      randomOrder.sort(() => Math.random() - 0.5);
      // setObjects(randomOrder);
      setTimeout(() => {
        objects.forEach((object) => {
          const element = document.getElementById(
            `item-${object.id}`
          );
          if (element) {
            element.style.transition =
              "all 0.48s cubic-bezier(0.23, 1, 0.32, 1)";
            element.style.opacity = "1"; // Fade in
            element.style.transform = "translate(0, 0)";
          }
        });
      }, 10);
    }, 300);
  };

  const distributeItems = (
    items: any[],
    pattern: number[]
  ) => {
    let result = [];
    let index = 0;
    let rowIndex = 0;
    while (index < items.length) {
      let row = [];
      let count = pattern[rowIndex % pattern.length];
      for (let i = 0; i < count; i++) {
        if (index < items.length) {
          row.push(items[index]);
          index++;
        }
      }
      result.push(row);
      rowIndex++;
    }
    return result;
  };

  const rowPattern = [2, 3, 2];
  const addItemPlaceholder = { id: null }; // Define the placeholder
  const itemsWithPlaceholder = [
    ...objects,
    addItemPlaceholder,
  ]; // Append the placeholder
  const rowItemsWithAddItem = distributeItems(
    itemsWithPlaceholder,
    rowPattern
  ); // Distribute normally

  // TODO implement flex grow from here: https://www.jiddo.ca/collection

  if (status === "loading") {
    return (
      <div className={styles["masonry-table-container"]}>
        <Divider />
        <div className={styles["loading"]}>
          <div className={styles["loading-spinner"]} />
        </div>
      </div>
    );
  }

  const getPlural = (count: number, singular: string, plural: string) => {
    // count ends in 1, but not 11
    if (count % 10 === 1 && count % 100 !== 11) {
      return singular;
    }
    return plural;
  }

  return (
    <div className={styles["masonry-table-container"]}>
      <Divider />
      <Row
        gutter={{ xs: 8, sm: 16, md: 24, lg: 32 }}
        style={{ width: "90%", margin: "0 auto" }}
      >
        {loading && (
          <div className={styles["loading"]}>
            <div className={styles["loading-spinner"]} />
          </div>
        )}

        {rowItemsWithAddItem.map((itemsInRow, rowIndex) => (
          <Row
            key={`row-${rowIndex}`}
            style={{ width: "100%", marginBottom: "2rem" }}
            gutter={[42, 62]}
          >
            {itemsInRow.map((item, itemIndex) => (
              <Col
                className={styles["content-wrapper"]}
                span={Math.min(24 / itemsInRow.length, 12)}
                key={`item-${item.id}`}
              >
                {item.id === null ? (
                  <div
                    onClick={handleAddButtonClick}
                    className={`${styles["content"]} ${styles["content-add"]}`}
                  >
                    <div
                      className={
                        styles["content-add-image"]
                      }
                    >
                      <PlusOutlined
                        style={{
                          fontSize: "48px",
                          color: "#fff",
                        }}
                      />
                    </div>
                    <span
                      className={
                        styles["content-add-title"]
                      }
                    >
                      Pievienot jaunu objektu
                    </span>
                  </div>
                ) : (
                  <div
                    id={`item-${item.id}`}
                    className={styles["content"]}
                    key={item.id}
                  >
                    {Boolean(
                      item.pictures &&
                      Array.isArray(item.pictures) &&
                      item.pictures.length
                    ) && (
                        <div
                          className={
                            styles["content-image-container"]
                          }
                        >
                          <div
                            className={
                              styles["content-image"]
                            }
                          >
                            <img
                              src={
                                item.pictures &&
                                  Array.isArray(
                                    item.pictures
                                  ) &&
                                  item.pictures.length > 0 &&
                                  typeof item.pictures[0] ===
                                  "object" &&
                                  "downloadUrl" in
                                  item.pictures[0]
                                  ? item.pictures[0]
                                    .downloadUrl
                                  : ""
                              }
                              alt="content"
                              style={{
                                height: "100%",
                                display: "block",
                              }}
                            />
                          </div>
                          {/* other images */}
                          <div
                            className={
                              styles["content-images-other"]
                            }
                          >
                            {item.pictures.length > 1 &&
                              item.pictures
                                .slice(1, 3)
                                .map(
                                  (
                                    picture: any,
                                    index: number
                                  ) => {
                                    return (
                                      <img
                                        key={index}
                                        src={
                                          picture &&
                                            typeof picture ===
                                            "object" &&
                                            "downloadUrl" in
                                            picture
                                            ? picture.downloadUrl
                                            : ""
                                        }
                                        alt="content"
                                      />
                                    );
                                  }
                                )}
                          </div>
                        </div>
                      )}

                    {/* TITLE */}
                    <div
                      className={
                        styles["content-title-wrapper"]
                      }
                    >
                      <span
                        className={`${styles["content-title-name"]
                          } ${item.pictures &&
                            Array.isArray(item.pictures) &&
                            item.pictures.length
                            ? ""
                            : styles[
                            "content-title-name-without-images"
                            ]
                          }`}
                      >
                        {item.name}
                      </span>
                    </div>

                    {/* DESCRIPTION */}

                    <div
                      className={
                        styles[
                        "content-description-wrapper"
                        ]
                      }
                    >
                      <div
                        className={
                          styles["content-description"]
                        }
                      >
                        {/* Header */}
                        <div
                          className={
                            styles[
                            "content-description-header"
                            ]
                          }
                        >
                          <div
                            className={
                              styles[
                              "content-description-header-left"
                              ]
                            }
                          >
                            <div className={styles["content-description-header-left-item"]}>
                              <span
                                className={
                                  styles[
                                  "content-description-header-left-item-title"
                                  ]
                                }
                              >
                                Adrese:
                              </span>
                              <span
                                className={
                                  styles[
                                  "content-description-header-left-item-value"
                                  ]
                                }
                                title={item.address ?? "N/A"}
                              >
                                {item.address ??
                                  "N/A"}
                              </span>
                            </div>
                          </div>
                          <div
                            className={
                              styles[
                              "content-description-header-right"
                              ]
                            }
                          >
                            <div
                              className={
                                styles["price-container"]
                              }
                            >
                              <span
                                className={
                                  styles["price-label"]
                                }
                              >
                                Tirgus cena:
                              </span>
                              <span
                                className={
                                  styles["price-value"]
                                }
                              >
                                {item.price ?? "N/A"} €
                              </span>
                            </div>
                            <div
                              className={
                                styles[
                                "predicted-price-container"
                                ]
                              }
                            >
                              <span
                                className={
                                  styles["price-label"]
                                }
                              >
                                Aprēķinātā cena:
                              </span>
                              <div
                                className={
                                  styles[
                                  "predicted-price-value-container"
                                  ]
                                }
                              >
                                <span
                                  className={
                                    styles["price-value"]
                                  }
                                >
                                  {item.predictedPrice > 0 ? (
                                    <>{item.predictedPrice} €</>
                                  ) : (
                                    <>
                                    
                                    <Tooltip title={
                                      "Cena šim objektam vēl tiek aprēķināta. Lūdzu, uzgaidiet."
                                    }>
                                      <IconButton>
                                        <QuestionMarkIcon style={{
                                          padding: "0",
                                          height: "1.1rem",
                                        }}/>
                                      </IconButton>
                                    </Tooltip>
                                      {/* <Loader style={{
                                        height: "1.1rem",
                                        display: "flex", 
                                        // outline: "1px solid red",
                                        width: "4rem",
                                      }} /> */}
                                    </>
                                    )
                                    }
                                </span>
                                {Boolean(item.predictedPrice) && (
                                  <>
                                    {item.predictedPrice > (item.price ?? 0) ? (
                                      <IoMdArrowRoundUp className={`${styles["price-arrow"]} ${styles["arrow-up"]}`} />
                                    ) : item.predictedPrice === (item.price ?? 0) ? (
                                      // <BiEqualizer className={`${styles["price-arrow"]} ${styles["arrow-neutral"]}`} />
                                      <></>
                                    ) : (
                                      <IoMdArrowRoundUp className={`${styles["price-arrow"]} ${styles["arrow-down"]}`} />
                                    )}
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div
                          className={
                            styles[
                            "content-description-toggle"
                            ]
                          }
                          onClick={() =>
                            toggleDescription(item.id)
                          }
                        >
                          <FaChevronDown
                            className={
                              openedDescriptions[item.id]
                                ? styles["icon-open"]
                                : ""
                            }
                          />
                          <span>
                            {openedDescriptions[item.id]
                              ? "Detalizēta informācija"
                              : "Detalizēta informācija"}
                          </span>
                        </div>

                        {/* Details list */}
                        <div
                          className={
                            styles[
                            "content-description-list"
                            ] +
                            (openedDescriptions[item.id]
                              ? ""
                              : ` ${styles["content-description-list-closed"]}`)
                          }
                        >
                          <li
                            className={
                              styles[
                              "content-description-list-item"
                              ]
                            }
                          >
                            <span
                              className={
                                styles[
                                "content-description-list-item-title"
                                ]
                              }
                            >
                              Pievienots:
                            </span>
                            <span
                              className={
                                styles[
                                "content-description-list-item-value"
                                ]
                              }
                            >
                              {new Intl.DateTimeFormat('lv-LV', {
                                day: 'numeric',
                                month: 'long',
                                year: 'numeric'
                              }).format(new Date(item.createdAt))}

                            </span>
                          </li>

                          <li
                            className={
                              styles[
                              "content-description-list-item"
                              ]
                            }
                          >
                            <span
                              className={
                                styles[
                                "content-description-list-item-title"
                                ]
                              }
                            >
                              Pieraksti:
                            </span>
                            <span
                              className={
                                styles[
                                "content-description-list-item-value"
                                ]
                              }
                            >
                              {item.description}
                            </span>
                          </li>
                          <li
                            className={
                              styles[
                              "content-description-list-item"
                              ]
                            }
                          >
                            <span
                              className={
                                styles[
                                "content-description-list-item-title"
                                ]
                              }
                            >
                              Platība:
                            </span>
                            <span
                              className={
                                styles[
                                "content-description-list-item-value"
                                ]
                              }
                            >
                              {`${item.area ?? "-"} m²`}
                            </span>
                          </li>
                        </div>

                        {/* Footer */}
                        <div
                          className={
                            styles[
                            "content-description-footer"
                            ] +
                            (openedDescriptions[item.id]
                              ? ""
                              : ` ${styles["content-description-footer-closed"]}`)
                          }
                        >
                          <div
                            className={
                              styles[
                              "content-description-house-details"
                              ]
                            }
                          >
                            <div
                              className={
                                styles[
                                "content-description-bed-counts"
                                ]
                              }
                            >
                              <IoBedOutline />
                              <span>
                                {`${item.bedroomCount ?? "-"} ${getPlural(item.bedroomCount, "Gulta", "Gultas")}`}
                              </span>
                            </div>
                            <div
                              className={
                                styles[
                                "content-description-bath-counts"
                                ]
                              }
                            >
                              <FaShower />
                              <span>
                                {`${item.bathroomCount ?? "-"} ${getPlural(item.bathroomCount, "Vannaistaba", "Vannaistabas")}`}
                              </span>
                            </div>
                            {Boolean(item.parkingCount) && (
                              <div
                                className={
                                  styles[
                                  "content-description-house-area"
                                  ]
                                }
                              >
                                <FaCarSide />
                              </div>
                            )}
                          </div>

                          <div
                            className={
                              styles[
                              "content-description-actions"
                              ]
                            }
                          >
                            <div
                              onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                onCardFavorite(item.id);
                              }}
                              className={`${styles[
                                "content-description-action"
                              ]
                                } ${styles[
                                "content-description-action-favourite"
                                ]
                                } ${item.favourite
                                  ? styles[
                                  "content-description-action-favourite-active"
                                  ]
                                  : ""
                                }`}
                            >
                              {item.favourite ? (
                                <HeartFilled />
                              ) : (
                                <HeartOutlined />
                              )}
                            </div>
                            <div
                              onClick={() =>
                                router.push(
                                  `/groups/${item.groupId}/${item.id}`
                                )
                              }
                              className={`${styles["content-description-action"]} ${styles["content-description-action-edit"]}`}
                            >
                              <EditOutlined />
                              Rediģēt
                            </div>

                            {showNavigateToGroup && (
                              <div
                                onClick={() =>
                                  router.push(
                                    `/groups/${item.groupId}`
                                  )
                                }
                                className={`${styles["content-description-action"]} ${styles["content-description-action-navigate"]}`}
                              >
                                <FaChevronUp />
                                Apskatīt grupu
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Col>
            ))}
          </Row>
        ))}
      </Row>

      {/* <div
          className={`${styles["content"]} ${styles["content-add"]}`}
          onClick={handleAddButtonClick}
        >
          <div className={styles["content-add-image"]}>
            <PlusOutlined style={{ fontSize: "48px", color: "#fff" }} />
          </div>
          <span className={styles["content-add-title"]}>
            Pievienot jaunu objektu
          </span>
        </div> */}
    </div>
  );
};

export default MasonryTable;
