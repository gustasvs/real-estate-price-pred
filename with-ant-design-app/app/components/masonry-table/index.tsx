"use client";

import {
  EditOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import QuestionMarkIcon from '@mui/icons-material/QuestionMark';
import {
  FaChevronDown,
  FaChevronUp,
  FaShower,
} from "react-icons/fa6";
import { IoBedOutline } from "react-icons/io5";
import { FaCarSide, FaDoorOpen, FaRegBuilding } from "react-icons/fa";
import {
  Divider,
  Button,
  Switch,
} from "antd";

import { IoMdArrowRoundUp } from "react-icons/io";


import styles from "./MasonryTable.module.css";
import { useRouter, useSearchParams } from "next/navigation";

import { useEffect, useState } from "react";

import {
  getObjects as getObjectsApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";
import { useSession } from "next-auth/react";
import { Tooltip } from "@mui/material";
import SearchInput from "../search-input/SearchInput";

import FavouriteButton from "./FavouriteButton";
import { StyledIconButton } from "../styled-mui-components/styled-components";

export const getPlural = (count: number, singular: string, plural: string) => {
  // count ends in 1, but not 11
  if (count % 10 === 1 && count % 100 !== 11) {
    return singular;
  }
  return plural;
}

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


const MasonryTable = ({
  group_id,
  columnCount,
  objects,
  showNavigateToGroup = false,
  revalidateDataFunction = () => { },
}: {
  group_id: string;
  columnCount: number;
  objects: any[];
  showNavigateToGroup?: boolean;
  revalidateDataFunction?: () => void;
}): JSX.Element => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, status, update } = useSession();

  const [localObjects, setLocalObjects] = useState(distributeItems(
    objects,
    [2, 3, 2]
  ));
  const [isLoading, setLoading] = useState(false);

  const [sortState, setSortState] = useState({ field: null, order: null } as { field: string | null, order: string | null });

  const [residenceName, setResidenceName] = useState("" as string);

  console.log("rerendering masonry table", objects);

  const onCardFavorite = async (id: string) => {
    const object = objects.find(
      (object) => object.id === id
    );
    if (!object) {
      return;
    }

    // lets assume it will be set successfully
    // setLocalObjects((prev) => {
    //   return prev.map((obj) => {
    //     if (obj.id === id) {
    //       return {
    //         ...obj,
    //         favourite: !obj.favourite,
    //       };
    //     }
    //     return obj;
    //   });
    // });
    setLocalObjects((prev) => {
      return prev.map((row) => {
        return row.map((obj) => {
          if (obj.id === id) {
            return {
              ...obj,
              favourite: !obj.favourite,
            };
          }
          return obj;
        });
      });
    });

    updateObjectApi(id, {
      favourite: !object.favourite,
    });
  };

  const [openedDescriptions, setOpenedDescriptions] =
    useState<{ [key: number]: boolean }>({});

  const [compactMode, setCompactMode] = useState(false);

  const toggleDescription = (id: number) => {
    setOpenedDescriptions((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const handleAddButtonClick = () => {
    if (group_id === "favourites") {
      router.push(`/groups`);
    } else {
      router.push(`/groups/${group_id}/new`);
    }
  };

  const fadeOut = (element: HTMLElement | null, fadeCompletely: boolean) => {
    if (!element) {
      return;
    }

    // const randomX = 10 * (Math.random() > 0.5 ? 1 : -1);
    // const randomY = 10 * (Math.random() > 0.5 ? 1 : -1);

    element.style.transition = fadeCompletely
      ? `all 0.1s ease-out` // Fast transition when fading completely
      : `all 0.78s cubic-bezier(0.23, 1, 0.32, 1)`;
    element.style.opacity = fadeCompletely ? "0" : "0.4";
    // element.style.transform = `translate(${randomX}px, ${randomY}px)`;
    element.style.transform = 'scale(0.96)';
    // element.style.filter = "blur(2px)";
  };

  const fadeIn = (element: HTMLElement | null) => {
    if (!element) {
      return;
    }
    element.style.transition = `all 0.88s cubic-bezier(0.23, 1, 0.32, 1)`;
    element.style.opacity = "1"; // Fade in after delay
    // element.style.transform = "translate(0, 0)";
    element.style.transform = 'scale(1)';
    // element.style.filter = "blur(0px)";
  };

  useEffect(() => {
    if (!isLoading) {
      objects.forEach((object) => {
        const element = document.getElementById(`item-${object.id}`);
        fadeIn(element);
      });
    } else {
      const objects = document.querySelectorAll(`.${styles["content"]}`);
      objects.forEach((object) => {
        fadeOut(object as HTMLElement, false);
      });
    }
  }, [isLoading]);

  useEffect(() => {
    console.log(sortState);
  }, [sortState]);

  const getNextSortOrder = (field: string) => {
    if (sortState.field !== field) return 'asc';
    if (sortState.order === 'asc') return 'desc';
    if (sortState.order === 'desc') return null;
    return 'asc';
  };

  const animateAndSort = async (field: string, fadeCompletely: boolean = false) => {

    const nextOrder = getNextSortOrder(field);
    setSortState({ field: nextOrder !== null ? field : null, order: nextOrder });


    // fade out all items
    objects.forEach((object) => {
      const element = document.getElementById(`item-${object.id}`);
      fadeOut(element, fadeCompletely);
    });
    // fadeOut(document.getElementById(`item-add`), fadeCompletely);

    // Set delay before reappearing
    const hiddenDuration = fadeCompletely ? 100 : 0; // Stay hidden for 1 second if fadeCompletely is true

    try {
      const sortedObjects = await getObjectsApi(group_id, { residenceName: residenceName, sortBy: field, sortOrder: nextOrder });
      if (!('error' in sortedObjects)) {
        setLocalObjects(distributeItems(sortedObjects, [2, 3, 2]));
      } else {
        console.error(sortedObjects.error);
      }
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch sorted objects:', error);
    }



    // fade back in

    setTimeout(() => {
      objects.forEach((object) => {
        const element = document.getElementById(`item-${object.id}`);
        fadeIn(element);
      });
      // fadeIn(document.getElementById(`item-add`));
    }, hiddenDuration); // Wait before reappearing
  };


  

  const rowPattern = [2, 3, 2];
  const addItemPlaceholder = { id: null }; // Define the placeholder
  const itemsWithPlaceholder = [
    // ...objects,
    ...localObjects,
    addItemPlaceholder,
  ]; // Append the placeholder
  // const rowItemsWithAddItem = distributeItems(
  //   itemsWithPlaceholder,
  //   rowPattern
  // ); // Distribute normally

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

  return (
    <div className={styles["masonry-table-container"]}>
      <div className={styles["masonry-table-header"]}>
        <SearchInput
          style={{
            marginTop: "0",
            width: "100%",
          }}
          placeholder="Meklēt objektu pēc tā nosaukuma..." 
          onChange={async (e: any) => {

            const value = e.target.value;

            // commented out because as of now (12/19/2024) nextjs cant handle route params without rerendering 
            // maybe in the future this can be implemented
            
            // const params = new URLSearchParams(searchParams);
            // if (value) {
            //   params.set('residenceName', value);
            // } else {
            //   params.delete('residenceName');
            // }
            // window.history.replaceState(null, '', `?${params.toString()}`);
            // const paramsObject = Object.fromEntries(params.entries());

            setLoading(true);

            setResidenceName(value);

            const newLocalObjects = await getObjectsApi(group_id, {residenceName: value, sortBy: sortState.field, sortOrder: sortState.order});

            if (!('error' in newLocalObjects)) {
              setLocalObjects(distributeItems(
                newLocalObjects,
                [2, 3, 2]
              ));
            } else {
              console.error(newLocalObjects.error);
            }

            setLoading(false);

            // router.replace(`?${params.toString()}`, { scroll: false });
          }}
        />
        <Button
          className={styles["sort-button"]}
          onClick={(e) => animateAndSort("createdAt", false)}
        >
          Kārtot pēc datuma
          <IoMdArrowRoundUp className={`${styles["sort-arrow"]} ${styles[`${sortState.field === "createdAt" ? sortState.order === "asc" ? "sort-arrow-up" : "sort-arrow-down" : "not-active"}`]}`} />
        </Button>
        <Button
          className={styles["sort-button"]}
          onClick={(e) => animateAndSort("price", false)}
        >
          Kārtot pēc cenas
          <IoMdArrowRoundUp className={`${styles["sort-arrow"]} ${styles[`${sortState.field === "price" ? sortState.order === "asc" ? "sort-arrow-up" : "sort-arrow-down" : "not-active"}`]}`} />
        </Button>
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "row",
          gap: "1rem",
          color: "var(--background-light-main)",
          backgroundColor: "var(--background-dark-main-hover)",
          border: "1px solid var(--background-light-secondary)",
          borderRadius: "1em",
          padding: "0 1rem",
        }}>
          <span
            style={{
              width: "fit-content",
              display: "flex",
              whiteSpace: "nowrap",
              fontSize: ".8rem",
              fontWeight: "600",
            }}
          >Kompaktais režīms:</span>
          <Switch
            // color="primary"
            className={styles["sort-switch"]}
            checked={compactMode}
            onChange={(changed) => {
              
              objects.forEach((object) => {
                const element = document.getElementById(`item-${object.id}`);
                fadeOut(element, true);
              });
              setCompactMode(changed);

              setTimeout(() => {
                objects.forEach((object) => {
                  const element = document.getElementById(`item-${object.id}`);
                  fadeIn(element);
                });
              }, 200);
            }} />
        </div>

        <Button
          type="primary"
          className={styles["create-new-button"]}
          icon={<PlusOutlined />}
          onClick={handleAddButtonClick}
        >
          Pievienot jaunu objektu
        </Button>

      </div>
      <div
        style={{ width: "90%", margin: "0 auto" }}
      >
        {/* {isLoading ? (
          <div className={styles["loading"]}>
            <div className={styles["loading-spinner"]} />
          </div>
        ) : (
          <> */}

        {localObjects.length === 0 && (
          <div>
            <div className={styles["no-objects"]}/>
            <span className={styles["no-objects-text"]}>
              Nav atrasts neviens objekts
            </span>
          </div>
        )}

        {localObjects.map((itemsInRow, rowIndex) => (
          <div
            key={`row-${rowIndex}`}
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "3rem",
              justifyContent: "space-between",
              marginBottom: "3rem",
              width: "100%",
            }}
          >
            {itemsInRow.map((item, itemIndex) => (
              <div
                className={`${styles["content-wrapper"]} ${!compactMode ? styles["content-wrapper-not-compact"] : ""}`}
                style={{
                  // flex: "40%",
                  flex: compactMode ? "40%" : "100%",
                }}
                key={`item-${item.id || "add"}`}
              >
                {item.id === null ? (
                  <div
                    id="item-add"
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
                    className={`${styles["content"]} ${!compactMode ? styles["content-not-compact"] : ""}`}
                    key={item.id}
                  >
                    {Boolean(
                      item.pictures &&
                      Array.isArray(item.pictures) &&
                      item.pictures.length
                    ) ? (
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
                      ) : (
                        // just so the base exists
                        <div
                          className={
                            styles["content-image-container"]
                          }
                        />
                      )
                    }

                    {/* TITLE */}
                    {compactMode ? (
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
                    ) : (
                      <span
                        className={`${styles["content-title-name"]
                          } ${styles["content-title-name-not-compact"]}`}
                      >
                        {item.name}
                      </span>
                    )
                    }

                    {/* DESCRIPTION */}

                    <div
                      className={`${styles["content-description-wrapper"]} ${!compactMode
                          ? styles["content-description-wrapper-not-compact"]
                          : ""
                        }`}
                    >
                      <div
                        className={
                          `${styles["content-description"]} ${!compactMode
                            ? styles["content-description-not-compact"]
                            : ""
                          }`
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
                                        <StyledIconButton>
                                          <QuestionMarkIcon style={{
                                            padding: "0",
                                            height: "1.1rem",
                                          }} />
                                        </StyledIconButton>
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

                        {compactMode && (
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
                        )}

                        {/* Details list */}
                        <div
                          className={`${styles["content-description-list"]} ${!openedDescriptions[item.id] && compactMode && styles["content-description-list-closed"]}`}
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
                              Piezīmes:
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
                          className={`${styles["content-description-footer"]} ${!openedDescriptions[item.id] && compactMode && styles["content-description-footer-closed"]}`}
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
                                "content-description-bath-counts"
                                ]
                              }
                            >
                              <FaDoorOpen />
                              <span>
                                {`${item.roomCount ?? "-"} ${getPlural(item.roomCount, "Istaba", "Istabas")}`}
                              </span>
                            </div>
                            <div
                              className={
                                styles[
                                "content-description-bed-counts"
                                ]
                              }
                            >
                              {/* <IoBedOutline /> */}
                              <FaRegBuilding className={styles["content-description-floor-icon"]} />
                              <span>
                                {/* {`${item.bedroomCount ?? "-"} ${getPlural(item.bedroomCount, "Stāvs", "Stāvi")}`} */}
                                {`${item.floor ?? "-"}/${item.buildingFloors ?? "-"} ${getPlural(item.floor, "Stāvs", "Stāvs")}`}
                              </span>
                            </div>
                            {Boolean(item.parkingAvailable) && (
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
                            <FavouriteButton
                              favourite={item.favourite}
                              onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                onCardFavorite(item.id);
                              }}
                            />
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
              </div>
            ))}
          </div>
        ))}
        {/* </>
      )} */}
      </div>
    </div>
  );
};

export default MasonryTable;
