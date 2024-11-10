import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Sidebar,
  Menu,
  MenuItem,
  SubMenu,
  sidebarClasses,
  menuClasses,
} from "react-pro-sidebar";
import Logo from "../navbar/Logo";

import styles from "./Sidebar.module.css";
import {
  HddOutlined,
  HeartOutlined,
  HomeOutlined,
  LeftSquareFilled,
  LeftSquareOutlined,
  PlusOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { getGroupsForSidebar } from "../../../../actions/group";
import { NextPageContext } from "next";
import NavLink from "../../NavLink/NavLink";
import {
  useParams,
  usePathname,
  useSearchParams,
} from "next/navigation";
import { Divider } from "antd";
import { FaGears, FaPersonFalling } from "react-icons/fa6";

const RewindUiSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  const pathname = usePathname();
  const params = useSearchParams();

  console.log("pathname", pathname, "params", params);

  const [groups, setGroups] = useState<any>(null);
  useEffect(() => {
    getGroupsForSidebar().then((groups) => {
      setGroups(groups);
    });
  }, []);

  return (
    <div
      style={{
        display: "flex",
        position: "relative",
      }}
    >
      <div
        className={`${styles[`left-sidebar-collapse`]}
      ${
        collapsed
          ? styles[`left-sidebar-collapse-collapsed`]
          : ``
      }
      `}
        onClick={() => setCollapsed(!collapsed)}
      >
        <LeftSquareFilled />
      </div>
      <Sidebar
        collapsed={collapsed}
        rootStyles={{
          [`.${menuClasses.subMenuRoot}`]: {
            borderRadius: "10px",
          },
          [`.${sidebarClasses.container}`]: {
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            paddingLeft: ".5rem",
          },
          [`.${menuClasses.button}`]: {
            borderRadius: "10px",
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            height: "3em",
            paddingLeft: "1em",
            a: {
              textWrap: "wrap",
            },
            "&:hover": {
              borderRadius: "10px",
              backgroundColor:
                "var(--background-dark-secondary) !important",
              color: "var(--background-light-secondary)",
            },
          },
          [`.${menuClasses.icon}`]: {
            svg: {
              fontSize: "0.5rem" // TODO not applied
            },
          },
          [`.${menuClasses.label}`]: {
            color: "inherit",
            backgroundColor: "inherit",
            textWrap: "wrap",
            "&:hover": {
              color: "inherit",
              backgroundColor: "inherit",
            },
          },
          [`.${menuClasses.icon}`]: {
            color: "inherit",
            backgroundColor: "inherit",
            "&:hover": {
              color: "inherit",
              backgroundColor: "inherit",
            },
          },
          [`.${menuClasses.menuItemRoot}`]: {
            borderRadius: "10px",
          },

          [`.${menuClasses.subMenuContent}`]: {
            marginLeft: "2rem",
            borderLeft:
              "1px solid var(--background-light-secondary)",
            backgroundColor: "var(--background-dark-main)",
          },

          [`.${menuClasses.SubMenuExpandIcon}`]: {
            paddingBottom: "10px",
          },

          [`.${menuClasses.active}`]: {
            backgroundColor: "var(--background-gray)",
            color: "var(--background-dark-main)",
            "&:hover": {
              backgroundColor:
                "var(--background-gray) !important",
              color: "var(--background-dark-main)",
            },

            [`.${menuClasses.button}`]: {
              backgroundColor: "var(--background-gray)",
              color: "var(--background-dark-main)",
              "&:hover": {
                backgroundColor:
                  "var(--background-light-secondary) !important",
                color: "var(--background-dark-main)",
              },
            },
            [`.${menuClasses.subMenuContent}`]: {
              [`.${menuClasses.menuItemRoot}`]: {
                backgroundColor: "var(--background-gray)",
                [`.${menuClasses.button}`]: {
                  borderRadius: "10px",
                },
                [`.${menuClasses.active}`]: {
                  backgroundColor:
                    "var(--background-light-secondary)",
                  color: "var(--background-dark-main)",
                  "&:hover": {
                    backgroundColor:
                      "var(--background-light-main) !important",
                    color: "var(--background-dark-main)",
                  },
                },
              },
            },
          },
        }}
      >
        <div
          className={styles["left-sidebar-company-logo"]}
        >
          <Logo />
          <span
            className={styles["left-sidebar-company-title"]}
          >
            Icn
          </span>
        </div>
        <Menu>
          <MenuItem
            icon={<HomeOutlined />}
            href="/"
            active={pathname === "/"}
          >
            Sākums
          </MenuItem>
          {/* </SubMenu> */}

          <SubMenu
            icon={<HddOutlined />}
            active={pathname.includes("/groups")}
            label={"Manas Grupas"}
          >
            {groups?.map((group: any) => (
              <MenuItem
                key={group.id}
                href={`/groups/${group.id}`}
                active={pathname === `/groups/${group.id}`}
              >
                {group.name}
              </MenuItem>
            ))}
            <MenuItem
              icon={<PlusOutlined />}
              href="/groups/new"
            >
              Pievienot jaunu
            </MenuItem>
          </SubMenu>

          <SubMenu
            label="Mans profils"
            icon={<UserOutlined />}
            active={pathname === "/profile"}
          >
            <MenuItem
              icon={<FaPersonFalling />}
              active={
                pathname === "/profile" &&
                (params.get("page") === "0" || !params.get("page"))
              }
              href="/profile?page=0"
            >
              {/* <NavLink href="/profile?page=0"> */}
              Lietotāja informācija
              {/* </NavLink>   */}
            </MenuItem>
            <MenuItem
              icon={<HeartOutlined />}
              active={
                pathname === "/profile" &&
                params.get("page") === "1"
              }
              href="/profile?page=1"
            >
              {/* <NavLink href="/profile?page=1"> */}
              Atzīmetas dzīvesvietas
              {/* </NavLink> */}
            </MenuItem>
            <MenuItem
              active={
                pathname === "/profile" &&
                params.get("page") === "2"
              }
              icon={<FaGears />}
              href="/profile?page=2"
            >
              {/* <NavLink href="/profile?page=2"> */}
              Iestatījumi
              {/* </NavLink> */}
            </MenuItem>
          </SubMenu>
        </Menu>
      </Sidebar>
    </div>
  );
};

export default RewindUiSidebar;
