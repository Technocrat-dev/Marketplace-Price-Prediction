'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Navbar.module.css';

const navLinks = [
    { href: '/', label: 'Predict' },
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/explore', label: 'Explore' },
    { href: '/about', label: 'About' },
];

export default function Navbar() {
    const pathname = usePathname();

    return (
        <nav className={styles.nav}>
            <div className={styles.inner}>
                <Link href="/" className={styles.logo}>
                    <span className={styles.logoIcon}>₹</span>
                    PriceScope
                </Link>

                <div className={styles.links}>
                    {navLinks.map(({ href, label }) => (
                        <Link
                            key={href}
                            href={href}
                            className={`${styles.link} ${pathname === href ? styles.linkActive : ''
                                }`}
                        >
                            {label}
                        </Link>
                    ))}
                </div>

                <div className={styles.status}>
                    <span className={styles.dot} />
                    Model Online
                </div>
            </div>
        </nav>
    );
}
