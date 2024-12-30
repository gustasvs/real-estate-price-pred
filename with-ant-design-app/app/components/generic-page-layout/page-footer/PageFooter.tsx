import styles from "./PageFooter.module.css";

import React from "react";
import {
  Breadcrumb,
  Divider,
  Layout,
  Menu,
  theme,
} from "antd";

import {
  FacebookOutlined,
  GithubOutlined,
  GitlabOutlined,
  InstagramOutlined,
  LeftSquareFilled,
  XOutlined,
  YoutubeOutlined,
} from "@ant-design/icons";

const PageFooter = () => {
  return (
    <div className={styles["footer-container"]}>
      <Divider style={{ borderColor: "var(--background-light-secondary)" }}>
        <div className={styles["footer-social"]}>
          {/* <FacebookOutlined
              className={styles["footer-social-icon"]}
            /> */}
          {/* <InstagramOutlined
              className={styles["footer-social-icon"]}
            /> */}
          {/* <YoutubeOutlined
            className={styles["footer-social-icon"]}
            onClick={() => {
              window.open(
                "https://www.youtube.com/c/gustasvs",
              );
            }
            }
          /> */}
          <XOutlined
            className={styles["footer-social-icon"]}
            // onClick={() => {
            //   "use client";
            //   window.open(
            //     "https://x.com/gustasvs",	
            //   );
            // }
            // }
          />
          <GithubOutlined
            className={styles["footer-social-icon"]}
            // onClick={() => {
            //   "use client";
            //   window.open(
            //     "https://github.com/gustasvs",
            //   );
            // }
            // }
          />
        </div>
      </Divider>
      <div className={styles["footer-logo"]}>
        <div
          className={styles["footer-logo-icon"]}
        />
        <div className={styles["footer-logo-text"]}>
          <span className={styles["footer-logo-title"]}>
            "SmartEstate"
          </span>
          © {new Date().getFullYear()}
        </div>
      </div>
      <div className={styles["footer-links"]}>
        <a href="/legal-stuff">Juridiskā informācija</a>
        <a href="/privacy-policy">Privātuma politika</a>
        <a href="/security">Drošība</a>
        <a href="/website-accessibility">
          Vietnes pieejamība
        </a>
        <a href="/manage-cookies">Pārvaldīt sīkdatnes</a>
      </div>
    </div>
  );
};

export default PageFooter;
