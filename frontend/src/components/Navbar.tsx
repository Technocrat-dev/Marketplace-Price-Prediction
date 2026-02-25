'use client';

import { useState } from 'react';
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
    const [mobileOpen, setMobileOpen] = useState(false);

    return (
        <nav className={styles.nav} aria-label="Main navigation">
            <div className={styles.inner}>
                <Link href="/" className={styles.logo} aria-label="PriceScope home">
                    <span className={styles.logoIcon}>₹</span>
                    PriceScope
                </Link>

                <button
                    className={styles.hamburger}
                    onClick={() => setMobileOpen(!mobileOpen)}
                    aria-label="Toggle navigation menu"
                    aria-expanded={mobileOpen}
                >
                    <span />
                    <span />
                    <span />
                </button>

                <div className={`${styles.links} ${mobileOpen ? styles.linksOpen : ''}`}>
                    {navLinks.map(({ href, label }) => (
                        <Link
                            key={href}
                            href={href}
                            className={`${styles.link} ${pathname === href ? styles.linkActive : ''}`}
                            onClick={() => setMobileOpen(false)}
                        >
                            {label}
                        </Link>
                    ))}
                </div>

                <div className={styles.status} aria-label="Model status">
                    <span className={styles.dot} aria-hidden="true" />
                    Model Online
                </div>
            </div>
        </nav>
    );
}
