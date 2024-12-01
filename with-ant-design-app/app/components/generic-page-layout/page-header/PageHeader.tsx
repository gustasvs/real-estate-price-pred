// PageHeader component with TypeScript support for breadcrumbs and a title

"use client";

import React from 'react';
import { Breadcrumb } from 'antd';
import { HomeOutlined } from '@ant-design/icons';

import styles from './PageHeader.module.css';
import { useRouter } from 'next/navigation';
import { auth } from '../../../../auth';
import { useSession } from 'next-auth/react';

type BreadcrumbItem = {
  label: string;
  path: string;
};

interface PageHeaderProps {
  title: string;
  breadcrumbItems: BreadcrumbItem[];
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, breadcrumbItems }) => {
  const router = useRouter();


  const { data: session, status, update } = useSession();

  if (status === 'loading') {
    return <div>Loading...</div>;
  }

  return (
    <div
      className={styles.pageHeader}
    >
      <Breadcrumb>
        <Breadcrumb.Item
          className={styles.breadcrumbItem}
          onClick={() => {
            router.push('/');
          }}
        >
          <HomeOutlined />
        </Breadcrumb.Item>
        {breadcrumbItems.map((item, index) => (
          <Breadcrumb.Item key={index}
          onClick={() => {
            router.push(item.path);
          }}
          className={styles.breadcrumbItem}
          >
            {/* <span> */}
              {item.label}

            {/* </span> */}
          </Breadcrumb.Item>
        ))}
      </Breadcrumb>
      <span className={styles.title}>{title}</span>
    </div>
  );
};

export default PageHeader;
