import type { ThemeConfig } from "antd";

const theme: ThemeConfig = {
  components: {
    Upload: {
      fontSize: 16,
      colorPrimary: "#35393b",
      colorTextSecondary: "#dddddd",
      borderRadiusLG: 16,
      fontFamily: "inherit"
      // width: 200,
    },
    Popover: {
      colorBgElevated: "var(--background-dark-secondary)",
      colorText: "var(--background-light-secondary)",
      fontFamily: "inherit",
    },
    Breadcrumb: {
      fontSize: 16,
      colorText: "var(--background-light-main)",
      colorTextSecondary: "var(--background-light-main)",
      separatorColor: "var(--background-light-main)",
      linkColor: "var(--background-light-main)",
      linkHoverColor: "var(--text-brighter)"
    },
    Button: {
      defaultHoverBg: "var(--background-dark-secondary)",
      defaultHoverColor: "var(--background-light-secondary)",
      colorPrimary: "var(--background-light-secondary)",
      colorPrimaryBgHover: "var(--background-dark-secondary)",
    }
  },
  token: {
    fontSize: 16,
    colorPrimary: "#35393b",
    colorTextSecondary: "#dddddd",
    borderRadiusLG: 16,
    fontFamily: "inherit"
  },
};

export default theme;
