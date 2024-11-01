"use client";

import {
  EditOutlined,
  HeartFilled,
  HeartOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import {
  FaChevronDown,
  FaChevronUp,
  FaShower,
} from "react-icons/fa6";
import { IoBedOutline } from "react-icons/io5";
import { FaCarSide, FaRuler } from "react-icons/fa";
import {
  Card,
  Row,
  Col,
  Button,
  Descriptions,
  Divider,
  Grid,
} from "antd";
import Image from "next/image";

import styles from "./MasonryTable.module.css";
import { useRouter } from "next/navigation";

import Layout from "react-masonry-list";
import { PageHeader } from "@ant-design/pro-components";
import { useState } from "react";
import { ResidenceObjectType } from "../../groups/[group_id]/page";

const MasonryTable = ({
  columnCount,
  onCardEdit = () => {},
  onCardFavorite = () => {},
  objects,
  createObject = () => {},
  deleteObject = () => {},
  updateObject = () => {},
  loading = false,
}: {
  columnCount: number;
  onCardEdit?: (id: string) => void;
  onCardFavorite?: (id: string) => void;
  objects: any[];
  createObject: () => void;
  deleteObject: (id: string) => void;
  updateObject: (
    id: string,
    objectData: ResidenceObjectType
  ) => void;
  loading?: boolean;
}): JSX.Element => {
  const [openedDescriptions, setOpenedDescriptions] =
    useState<{ [key: number]: boolean }>({});

  const toggleDescription = (id: number) => {
    setOpenedDescriptions((prev) => ({
      ...prev,
      [id]: !prev[id], // Toggle the state for the specific id
    }));
  };

  const handleAddButtonClick = () => {
    createObject();
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

  return (
    <div className={styles["masonry-table-container"]}>
      <Divider />
      {/* <div className={styles["masonry-table"]}> */}
      <Row
        gutter={{ xs: 8, sm: 16, md: 24, lg: 32 }}
        style={{ width: "90%", margin: "0 auto" }}
      >
        {rowItemsWithAddItem.map((itemsInRow, rowIndex) => (
          <Row
            key={`row-${rowIndex}`}
            style={{ width: "100%" }}
            gutter={[42, 32]}
          >
            {itemsInRow.map((item, itemIndex) => (
              <Col
                className={styles["content-wrapper"]}
                span={Math.min(24 / itemsInRow.length, 12)}
                key={item.id}
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
                    style={{
                      marginBottom: 20, // Maintain bottom margin for spacing
                    }}
                  >
                    <div
                      className={
                        styles["content-image-container"]
                      }
                    >
                      <div
                        className={styles["content-image"]}
                      >
                        <img
                          src={
                            item.pictures[0]?.startsWith(
                              "data:image"
                            )
                              ? item.pictures[0]
                              : `data:image/png;base64,${item.pictures[0]}`
                          }
                          alt="content"
                          style={{
                            height: "100%", // Ensure image matches the container's height
                            // width: "auto",
                            display: "block", // Remove any inline image spacing issues
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
                                picture: string,
                                index: number
                              ) => {
                                return (
                                  <img
                                    src={
                                      picture?.startsWith(
                                        "data:image"
                                      )
                                        ? picture
                                        : `data:image/png;base64,${picture}`
                                    }
                                    alt="content"
                                  />
                                );
                              }
                            )}
                      </div>
                    </div>

                    {/* TITLE */}
                    <div
                      className={
                        styles["content-title-wrapper"]
                      }
                    >
                      <span
                        className={
                          styles["content-title-name"]
                        }
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
                                "content-description-header-top"
                              ]
                            }
                          >
                            <span
                              className={
                                styles[
                                  "content-description-header-name"
                                ]
                              }
                            >
                              {item.name}
                            </span>
                            <span
                              className={
                                styles[
                                  "content-description-header-price"
                                ]
                              }
                            >
                              {item.price ??
                                "Nav norādīta cena"}
                            </span>
                          </div>
                          <div
                            className={
                              styles[
                                "content-description-header-bottom"
                              ]
                            }
                          >
                            <span
                              className={
                                styles[
                                  "content-description-header-date"
                                ]
                              }
                            >
                              Pievienots{" "}
                              {new Date(
                                item.createdAt
                              ).toLocaleDateString()}
                            </span>
                            <span
                              className={
                                styles[
                                  "content-description-header-price-prediction"
                                ]
                              }
                            >
                              {item.pricePrediction ??
                                "Nav norādīta cenu prognoze"}
                            </span>
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
                              Adrese:
                            </span>
                            <span
                              className={
                                styles[
                                  "content-description-list-item-value"
                                ]
                              }
                            >
                              {item.address ??
                                "Rīga, Jaunā iela 1 - 22"}
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
                              {`${item.area ?? 100} m²`}
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
                                {item.bedroomCount ??
                                  "3 Gultas"}
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
                                {item.bathroomCount ??
                                  "2 Dušas"}
                              </span>
                            </div>
                            <div
                              className={
                                styles[
                                  "content-description-house-area"
                                ]
                              }
                            >
                              <FaCarSide />
                              <span>{`${
                                item.parkingCount ?? "2"
                              } Stāvvietas`}</span>
                            </div>
                          </div>

                          <div
                            className={
                              styles[
                                "content-description-actions"
                              ]
                            }
                          >
                            <Button
                              type="primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                onCardFavorite(item.id);
                              }}
                              className={`${styles["content-description-action"]} ${styles["content-description-action-favourite"]} ${item.isFavourite ? styles["content-description-action-favourite-active"] : ""}`}
                            >
                              {item.isFavourite ? (
                                <HeartFilled />
                              ) : (
                                <HeartOutlined />
                              )}
                            </Button>
                            <Button
                              type="primary"
                              onClick={() =>
                                onCardEdit(item.id)
                              }
                              className={`${styles["content-description-action"]} ${styles["content-description-action-edit"]}`}
                            >
                              <EditOutlined />
                              Rediģēt
                            </Button>
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
