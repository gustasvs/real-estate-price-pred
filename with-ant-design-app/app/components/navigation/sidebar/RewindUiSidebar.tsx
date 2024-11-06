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
import { useParams, usePathname, useSearchParams } from "next/navigation";
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
          [`.${sidebarClasses.container}`]: {
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            paddingLeft: ".5rem",
          },
          [`.${menuClasses.button}`]: {
            color: "var(--background-light-main)",
            backgroundColor: "var(--background-dark-main)",
            height: "3em",
            paddingLeft: "1em",
            "a": {
              textWrap: "wrap",
            },

            "&:hover": {
              borderRadius: "10px",
              backgroundColor:
                "var(--background-dark-main-hover) !important",
              color: "var(--background-light-main)",
            },
          },
          [`.${menuClasses.active}`]: {
            backgroundColor: "var(--background-light-main)",
            color: "var(--background-dark-main)",
            borderRadius: "10px",
            "&:hover": {
              backgroundColor:
                "var(--background-light-secondary) !important",
              color: "var(--background-dark-main)",
            },
            "&:hover span": {
              backgroundColor:
                "var(--background-light-secondary) !important",
              color: "var(--background-dark-main)",
            },
          },
          [`.${menuClasses.label}`]: {
            // color: "inherit",
            backgroundColor: "inherit",
            "&:hover": {
              color: "inherit",
              backgroundColor: "inherit",
            },
          },

          [`.${menuClasses.subMenuContent}`]: {
            marginLeft: "2rem",
            borderLeft: "1px solid var(--background-light-secondary)",
            backgroundColor:
              "var(--background-dark-main)",
          },

          [`.${menuClasses.SubMenuExpandIcon}`]: {
            paddingBottom: "10px"
          }
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
            active={pathname === "/groups"}
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
          >
            <MenuItem
              icon={<FaPersonFalling />}
              active={pathname === "/profile"&& params.get("page") === "0"}
              href="/profile?page=0"
            >
              {/* <NavLink href="/profile?page=0"> */}
              Lietotāja informācija
              {/* </NavLink>   */}
              
            </MenuItem>
            <MenuItem
              icon={<HeartOutlined />}
              active={pathname === "/profile" && params.get("page") === "1"}
              href="/profile?page=1"
            >
              {/* <NavLink href="/profile?page=1"> */}
              Atzīmetas dzīvesvietas
              {/* </NavLink> */}
            </MenuItem>
            <MenuItem
              active={pathname === "/profile" && params.get("page") === "2"}
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
