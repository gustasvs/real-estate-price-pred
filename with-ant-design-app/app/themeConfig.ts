import type { ThemeConfig } from "antd";

const theme: ThemeConfig = {
  components: {
    Breadcrumb: {
      fontSize: 16,
      colorText: "var(--background-light-secondary)",
      colorTextSecondary: "var(--background-light-secondary)",
      separatorColor: "var(--background-light-secondary)",
    },
  },
  token: {
    fontSize: 16,
    colorPrimary: "#35393b",
    colorTextSecondary: "#dddddd",
    borderRadiusLG: 16,
  },
};

export default theme;
