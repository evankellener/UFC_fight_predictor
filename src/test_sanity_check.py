from sanity_check import sanity_check

success, results = sanity_check('../data/tmp/final.csv', 'Alexandre Pantoja')

if success:
    print("SANITY CHECK PASSED: All postcomp stats match the precomp stats in subsequent fights.")
else:
    print("SANITY CHECK FAILED: Some stats don't match between fights.")
    for fight, mismatch in results.items():
        print(f"\n{fight}:")
        print(f"  Stat: {mismatch['stat']}")
        print(f"  Postcomp value: {mismatch['postcomp_stat']}")
        print(f"  Precomp value: {mismatch['precomp_stat']}")
