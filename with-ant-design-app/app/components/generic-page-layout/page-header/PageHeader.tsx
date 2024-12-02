// PageHeader component with TypeScript support for breadcrumbs and a title

"use client";

import React from "react";
import { Breadcrumb } from "antd";
import { HomeOutlined } from "@ant-design/icons";

import styles from "./PageHeader.module.css";
import { useRouter } from "next/navigation";
import { auth } from "../../../../auth";
import { useSession } from "next-auth/react";

type BreadcrumbItem = {
  label: string;
  path: string;
};

interface PageHeaderProps {
  title: string;
  breadcrumbItems: BreadcrumbItem[];
}

const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  breadcrumbItems,
}) => {
  const router = useRouter();

  const { data: session, status, update } = useSession();

  if (status === 'loading') {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}
      >
        {/* Breadcrumb Placeholder */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            gap: '0.5em',
          }}
        >
          <span
            style={{
              display: 'inline-block',
              width: '16px',
              height: '16px',
              backgroundColor: 'var(--background-gray)',
              borderRadius: '50%',
            }}
          ></span>
          <span
            style={{
              display: 'inline-block',
              width: '80px',
              height: '14px',
              backgroundColor: 'var(--background-gray)',
              borderRadius: '4px',
            }}
          ></span>
          <span
            style={{
              margin: '0 0.5em',
              color: 'var(--background-gray)',
            }}
          >
            /
          </span>
          <span
            style={{
              display: 'inline-block',
              width: '120px',
              height: '14px',
              backgroundColor: 'var(--background-gray)',
              borderRadius: '4px',
            }}
          ></span>
          <span
            style={{
              margin: '0 0.5em',
              color: 'var(--background-gray)',
            }}
          >
            /
          </span>
          <span
            style={{
              display: 'inline-block',
              width: '60px',
              height: '14px',
              backgroundColor: 'var(--background-gray)',
              borderRadius: '4px',
            }}
          ></span>
        </div>
  
        {/* Title Placeholder */}
        <div
          style={{
            display: 'inline-block',
            width: '35rem',
            height: '3rem',
            backgroundColor: 'var(--background-gray)',
            borderRadius: '5px',
          }}
        ></div>
      </div>
    );
  }

  return (
    <div className={styles.pageHeader}>
      <Breadcrumb>
        <Breadcrumb.Item
          className={styles.breadcrumbItem}
          onClick={() => {
            router.push("/");
          }}
        >
          <HomeOutlined />
        </Breadcrumb.Item>
        {breadcrumbItems.map((item, index) => (
          <Breadcrumb.Item
            key={index}
            onClick={() => {
              if (index !== breadcrumbItems.length - 1) {
                router.push(item.path);
              }
            }}
            className={`${styles.breadcrumbItem} ${
              index === breadcrumbItems.length - 1
                ? styles.lastBreadcrumbItem
                : ""
            }`}
          >
            {item.label}
          </Breadcrumb.Item>
        ))}
      </Breadcrumb>
      <span className={styles.title}>{title}</span>
    </div>
  );
};

export default PageHeader;
